//! The bundled, offline model catalog.
//!
//! `catalog.json` is generated at build time by `scripts/gen_catalog.py` from the
//! `handy-computer` Hugging Face org (card `transcribe_cpp` capabilities +
//! benchmarks, a GGUF header probe for name/params, and local curation for the
//! recommended set). It is compiled into the binary so Handy ships a complete
//! model list with zero network access.
//!
//! `fork_catalog.json` contains hand-curated additions for this fork. Keeping
//! them separate prevents upstream catalog regeneration from removing them.
//! Mirror configuration belongs to each catalog, not to the merged model list.
//!
//! Each entry is normalised into a [`ModelDescriptor`] — the same source-agnostic
//! shape every other producer (HF discovery, on-disk scans, the legacy table)
//! yields — so the catalog is "just another producer". Its explicit `capabilities`
//! map becomes a [`CapabilityProbe`] with confident `Some(..)` values; the runtime
//! `GgufHeaderProber` is the same shape with `None` where a header omits a key,
//! which is why the two are interchangeable (the catalog is a baked probe).

use std::collections::HashMap;

use once_cell::sync::Lazy;
use serde::Deserialize;

use crate::managers::model::{
    default_quant_file, EngineType, ModelDescriptor, ModelSource, QuantFile,
};
use crate::managers::model_capabilities::{CapabilityProbe, Compatibility};

#[derive(Deserialize)]
struct CatalogRoot {
    /// Base URLs tried in order when the Hugging Face download fails. The full
    /// file URL is `{mirror}/{repo_id}/{revision}/{filename}` — the same three
    /// values that form the HF resolve URL, so a mirror is a plain static host.
    #[serde(default)]
    mirrors: Vec<String>,
    models: Vec<CatalogModel>,
}

/// One model as written in `catalog.json`. Only the fields the descriptor needs
/// are declared; serde ignores the rest (slug, family, license, …).
#[derive(Deserialize)]
struct CatalogModel {
    /// HF repo id, e.g. `handy-computer/whisper-small-gguf`.
    id: String,
    /// Commit sha the catalog's sizes/hashes were generated from. Both HF
    /// acquisition and mirror keys use it, so downloaded bytes provably match
    /// the hashes regardless of source. Cache *lookup* additionally falls back
    /// to `main` (see `hf_cached_path`) so downloads that predate pinning keep
    /// resolving.
    revision: Option<String>,
    name: String,
    description: String,
    architecture: Option<String>,
    languages: Vec<String>,
    capabilities: CatalogCaps,
    speed_score: Option<f32>,
    accuracy_score: Option<f32>,
    files: Vec<QuantFile>,
    default_quant: Option<String>,
    recommended_rank: Option<u32>,
    /// Part of the small curated onboarding set (badged "Recommended"). Distinct
    /// from `recommended_rank`, which only orders the full list.
    #[serde(default)]
    recommended: bool,
}

#[derive(Deserialize)]
struct CatalogCaps {
    streaming: bool,
    translate: bool,
    lang_detect: bool,
    // `timestamps` (a string enum) is present in the catalog but has no
    // `CapabilityProbe` field yet — wire it through when the probe gains one.
}

impl From<&CatalogModel> for ModelDescriptor {
    fn from(m: &CatalogModel) -> Self {
        // The default download file. Its name is folded into the id so a catalog
        // entry collides (dedups) with the very same file later discovered in
        // the HF cache — both compute `"{repo_id}/{filename}"`.
        let default_filename = default_quant_file(&m.files, m.default_quant.as_deref())
            .map(|f| f.filename.clone())
            .unwrap_or_default();

        ModelDescriptor {
            id: format!("{}/{}", m.id, default_filename),
            source: ModelSource::HuggingFace {
                repo_id: m.id.clone(),
                // Acquire at the pin: `resolve/<sha>` is immutable (CDN-friendly)
                // and guarantees the bytes match the catalog's hashes. `main`
                // only remains as a lookup fallback for pre-pinning caches.
                revision: m.revision.clone().unwrap_or_else(|| "main".to_string()),
            },
            name: m.name.clone(),
            description: m.description.clone(),
            engine_type: EngineType::TranscribeCpp,
            caps: CapabilityProbe {
                verdict: Compatibility::Compatible, // curated models we ship support for
                display_name: None,
                architecture: m.architecture.clone(),
                variant: None,
                languages: Some(m.languages.clone()),
                supports_streaming: Some(m.capabilities.streaming),
                supports_translation: Some(m.capabilities.translate),
                supports_language_detect: Some(m.capabilities.lang_detect),
            },
            files: m.files.clone(),
            default_quant: m.default_quant.clone(),
            // catalog scores are 0–100; ModelInfo / the UI bars use 0.0–1.0.
            speed_score: m.speed_score.unwrap_or(0.0) / 100.0,
            accuracy_score: m.accuracy_score.unwrap_or(0.0) / 100.0,
            recommended_rank: m.recommended_rank,
            recommended: m.recommended,
        }
    }
}

/// The raw parsed catalog. Kept alive (not consumed) so mirror metadata that
/// deliberately stays out of [`ModelDescriptor`] can be looked up separately.
static ROOT: Lazy<CatalogRoot> = Lazy::new(|| {
    serde_json::from_str(include_str!("catalog.json"))
        .expect("bundled catalog.json is valid JSON matching the catalog schema")
});

static FORK_ROOT: Lazy<CatalogRoot> = Lazy::new(|| {
    serde_json::from_str(include_str!("fork_catalog.json"))
        .expect("bundled fork_catalog.json is valid JSON matching the catalog schema")
});

fn catalog_roots() -> [&'static CatalogRoot; 2] {
    [Lazy::force(&ROOT), Lazy::force(&FORK_ROOT)]
}

/// The bundled catalog, parsed once and normalised into descriptors.
pub static CATALOG: Lazy<Vec<ModelDescriptor>> = Lazy::new(|| {
    catalog_roots()
        .into_iter()
        .flat_map(|root| root.models.iter().map(ModelDescriptor::from))
        .collect()
});

/// A mirror copy of a catalog model's default file, with the expected content
/// hash for end-to-end verification. Mirrors are untrusted bit-pipes: the
/// sha256 here (from the catalog compiled into the binary) is the trust anchor,
/// which is why it is mandatory — a file without one is never offered from a
/// mirror at all.
pub struct MirrorFile {
    pub url: String,
    pub sha256: String,
    /// Catalog size — drives progress totals and resume sanity checks.
    pub size_bytes: u64,
}

/// Ordered mirror URLs for a catalog model's file — any listed quant, not just
/// the default — or empty when the model isn't from the catalog / no mirrors
/// are configured. `model_id` is the registry id (`"{repo_id}/{filename}"`).
/// (The mirror may only host default quants; a miss there just 404s and the
/// caller reports it, so listing every quant here costs nothing.)
pub fn mirror_fallbacks(model_id: &str) -> Vec<MirrorFile> {
    let Some((root, m, file)) = catalog_roots().into_iter().find_map(|root| {
        root.models.iter().find_map(|m| {
            m.files
                .iter()
                .find(|f| format!("{}/{}", m.id, f.filename) == model_id)
                .map(|f| (root, m, f))
        })
    }) else {
        return Vec::new();
    };
    let Some(revision) = m.revision.as_deref() else {
        return Vec::new();
    };
    // No hash means no verification means no mirror: never fetch from an
    // untrusted host without the catalog trust anchor.
    let Some(sha256) = file.sha256.as_deref() else {
        return Vec::new();
    };
    root.mirrors
        .iter()
        .map(|base| MirrorFile {
            url: format!(
                "{}/{}/{}/{}",
                base.trim_end_matches('/'),
                m.id,
                revision,
                file.filename
            ),
            sha256: sha256.to_string(),
            size_bytes: file.size_bytes,
        })
        .collect()
}

/// The catalog descriptor + specific `files[]` entry owning `filename`,
/// matched across every listed quant (not just the default). `repo_id`, when
/// given, must also match — the HF-cache scan uses it to keep a foreign repo
/// that happens to reuse a catalog filename from masquerading as ours.
pub fn file_in_catalog(
    filename: &str,
    repo_id: Option<&str>,
) -> Option<(&'static ModelDescriptor, &'static QuantFile)> {
    let catalog: &'static Vec<ModelDescriptor> = Lazy::force(&CATALOG);
    catalog.iter().find_map(|d| {
        if let Some(repo) = repo_id {
            match &d.source {
                ModelSource::HuggingFace { repo_id: r, .. } if r == repo => {}
                _ => return None,
            }
        }
        d.files
            .iter()
            .find(|f| f.filename == filename)
            .map(|f| (d, f))
    })
}

/// Editorial recommended rank keyed by descriptor id (the same id the model
/// registry uses). Built once from the catalog.
static RANK_BY_ID: Lazy<HashMap<String, u32>> = Lazy::new(|| {
    CATALOG
        .iter()
        .filter_map(|d| d.recommended_rank.map(|r| (d.id.clone(), r)))
        .collect()
});

/// Recommended rank for a model id (lower = higher priority). Returns
/// `u32::MAX` for unranked/unknown ids so they sort last in an ascending sort.
pub fn rank_of(model_id: &str) -> u32 {
    RANK_BY_ID.get(model_id).copied().unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::managers::model_capabilities::KNOWN_ARCHES;
    use std::collections::BTreeSet;

    #[test]
    fn catalog_parses_and_is_nonempty() {
        assert!(!CATALOG.is_empty(), "bundled catalog should contain models");
    }

    #[test]
    fn ids_are_unique() {
        let mut ids: Vec<&str> = CATALOG.iter().map(|d| d.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "catalog descriptor ids must be unique");
    }

    #[test]
    fn pure_diarization_models_are_not_downloadable() {
        assert!(
            CATALOG
                .iter()
                .all(|model| model.caps.architecture.as_deref() != Some("sortformer")),
            "Sortformer produces speaker segments, not transcription text"
        );
    }

    #[test]
    fn scores_are_normalised_0_to_1() {
        for d in CATALOG.iter() {
            assert!((0.0..=1.0).contains(&d.speed_score), "{} speed", d.id);
            assert!((0.0..=1.0).contains(&d.accuracy_score), "{} acc", d.id);
        }
    }

    #[test]
    fn every_upstream_catalog_model_has_mirror_fallbacks_with_hashes() {
        // The mirror fallback is the safety net for HF outages and blocked
        // networks; a catalog entry without one (missing revision, missing
        // sha256, empty mirrors) silently loses that net.
        for d in ROOT.models.iter().map(ModelDescriptor::from) {
            let mirrors = mirror_fallbacks(&d.id);
            assert!(!mirrors.is_empty(), "{}: no mirror fallbacks", d.id);
            for m in &mirrors {
                assert!(
                    m.sha256.len() == 64,
                    "{}: mirror entry lacks a sha256",
                    d.id
                );
                assert!(m.size_bytes > 0, "{}: mirror entry lacks a size", d.id);
                assert!(m.url.starts_with("https://"), "{}: bad url {}", d.id, m.url);
            }
        }
    }

    #[test]
    fn parakeet_ultra_is_downloadable_with_pinned_verified_source() {
        use crate::managers::model::DiskStatus;

        let (descriptor, file) = file_in_catalog(
            "parakeet-ultra-0.6b-Q8_0.gguf",
            Some("Nairod785/parakeet-ultra-gguf"),
        )
        .expect("Parakeet Ultra must be in the downloadable catalog");
        assert_eq!(descriptor.default_quant.as_deref(), Some("Q8_0"));
        assert!(matches!(descriptor.engine_type, EngineType::TranscribeCpp));
        assert!(matches!(
            &descriptor.source,
            ModelSource::HuggingFace { repo_id, revision }
                if repo_id == "Nairod785/parakeet-ultra-gguf"
                    && revision == "b03613ba54a195238f0e915359f5a5c78269ddc6"
        ));
        assert_eq!(file.size_bytes, 739_508_704);
        assert_eq!(
            file.sha256.as_deref(),
            Some("283562ac9b513f39244fe23c6632738c167d32731a5f4693319a10ca498550a8")
        );

        let info = descriptor.to_model_info(&DiskStatus::default());
        assert_eq!(info.name, "Parakeet Ultra");
        assert_eq!(info.filename, file.filename);
        assert!(!info.is_custom);
        assert!(!info.is_downloaded);
        assert!(!info.is_directory);
        assert!(!info.supports_streaming);
        assert!(!info.supports_translation);
        assert!(info.supports_language_detection);
        assert!(info.supports_language_selection);
        assert_eq!(info.supported_languages.len(), 25);
        assert!(info.supported_languages.iter().any(|lang| lang == "es"));
        // Unknown comparative benchmarks must not become fabricated UI scores.
        assert_eq!(info.speed_score, 0.0);
        assert_eq!(info.accuracy_score, 0.0);
    }

    #[test]
    fn fork_models_do_not_inherit_upstream_mirrors() {
        for model in &FORK_ROOT.models {
            for file in &model.files {
                assert!(mirror_fallbacks(&format!("{}/{}", model.id, file.filename)).is_empty());
                assert!(file.size_bytes > 0);
                assert_eq!(file.sha256.as_deref().map(str::len), Some(64));
            }
        }
    }

    #[test]
    fn parakeet_ultra_alternate_quants_keep_catalog_metadata() {
        use crate::managers::model::DiskStatus;

        let filename = "parakeet-ultra-0.6b-Q4_K_M.gguf";
        let (descriptor, file) = file_in_catalog(filename, Some("Nairod785/parakeet-ultra-gguf"))
            .expect("alternate quants must be recognised on disk");
        let info = descriptor.to_model_info_for_file(
            file,
            &DiskStatus {
                is_downloaded: true,
                ..Default::default()
            },
        );
        assert_eq!(info.name, "Parakeet Ultra (Q4_K_M)");
        assert_eq!(info.id, format!("Nairod785/parakeet-ultra-gguf/{filename}"));
        assert!(info.is_downloaded);
        assert!(!info.is_custom);
        assert!(file_in_catalog(filename, Some("unrelated/repo")).is_none());
    }

    #[test]
    fn catalog_architectures_are_known_to_capability_probe() {
        let missing: BTreeSet<&str> = CATALOG
            .iter()
            .filter_map(|d| d.caps.architecture.as_deref())
            .filter(|arch| !KNOWN_ARCHES.contains(arch))
            .collect();

        assert!(
            missing.is_empty(),
            "catalog architecture(s) missing from KNOWN_ARCHES: {:?}",
            missing
        );
    }
}
