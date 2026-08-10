use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::protocol::{ModelMetadata, TensorMetadata};
use crate::{Error, ErrorCode, Result};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    pub name: String,
    pub platform: String,
    pub backend: String,
    pub max_batch_size: usize,
    pub max_queue_delay_us: u64,
    pub queue_capacity: usize,
    #[serde(default = "default_timeout_us")]
    pub default_timeout_us: u64,
    #[serde(default)]
    pub llm: bool,
    #[serde(default = "default_cache_blocks")]
    pub cache_blocks: usize,
    #[serde(default = "default_cache_block_tokens")]
    pub cache_block_tokens: usize,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

const fn default_timeout_us() -> u64 {
    5_000_000
}

const fn default_cache_blocks() -> usize {
    256
}

const fn default_cache_block_tokens() -> usize {
    16
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelKey {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug)]
pub struct LoadedModel {
    pub key: ModelKey,
    pub config: ModelConfig,
    pub directory: PathBuf,
}

#[derive(Debug, Default)]
struct RepositoryState {
    generation: u64,
    models: BTreeMap<ModelKey, LoadedModel>,
}

#[derive(Debug)]
pub struct ModelRepository {
    root: PathBuf,
    state: RwLock<RepositoryState>,
}

impl ModelRepository {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let repository = Self {
            root: root.as_ref().to_path_buf(),
            state: RwLock::new(RepositoryState::default()),
        };
        repository.reload()?;
        Ok(repository)
    }

    pub fn reload(&self) -> Result<u64> {
        let discovered = discover_models(&self.root)?;
        let mut state = self.state.write().expect("repository lock poisoned");
        state.generation = state.generation.saturating_add(1);
        state.models = discovered;
        Ok(state.generation)
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.state
            .read()
            .expect("repository lock poisoned")
            .generation
    }

    pub fn resolve(&self, name: &str, version: Option<&str>) -> Result<LoadedModel> {
        let state = self.state.read().expect("repository lock poisoned");
        if let Some(version) = version {
            return state
                .models
                .get(&ModelKey {
                    name: name.to_owned(),
                    version: version.to_owned(),
                })
                .cloned()
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::NotFound,
                        format!("model {name} version {version} not found"),
                    )
                });
        }
        state
            .models
            .values()
            .filter(|model| model.key.name == name)
            .max_by(|left, right| compare_versions(&left.key.version, &right.key.version))
            .cloned()
            .ok_or_else(|| Error::new(ErrorCode::NotFound, format!("model {name} not found")))
    }

    #[must_use]
    pub fn list(&self) -> Vec<LoadedModel> {
        self.state
            .read()
            .expect("repository lock poisoned")
            .models
            .values()
            .cloned()
            .collect()
    }

    pub fn metadata(&self, name: &str) -> Result<ModelMetadata> {
        let models = self
            .list()
            .into_iter()
            .filter(|model| model.key.name == name)
            .collect::<Vec<_>>();
        let latest = models
            .iter()
            .max_by(|left, right| compare_versions(&left.key.version, &right.key.version))
            .ok_or_else(|| Error::new(ErrorCode::NotFound, format!("model {name} not found")))?;
        Ok(ModelMetadata {
            name: name.to_owned(),
            versions: models
                .iter()
                .map(|model| model.key.version.clone())
                .collect(),
            platform: latest.config.platform.clone(),
            inputs: latest.config.inputs.clone(),
            outputs: latest.config.outputs.clone(),
        })
    }
}

fn discover_models(root: &Path) -> Result<BTreeMap<ModelKey, LoadedModel>> {
    if !root.exists() {
        return Err(Error::new(
            ErrorCode::NotFound,
            format!("model repository {} does not exist", root.display()),
        ));
    }
    let mut models = BTreeMap::new();
    for model_entry in sorted_directories(root)? {
        let model_name = model_entry
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::new(ErrorCode::InvalidArgument, "non-UTF8 model name"))?
            .to_owned();
        for version_entry in sorted_directories(&model_entry)? {
            let version = version_entry
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| Error::new(ErrorCode::InvalidArgument, "non-UTF8 version"))?
                .to_owned();
            let config_path = version_entry.join("model.json");
            if !config_path.is_file() {
                continue;
            }
            let config: ModelConfig = serde_json::from_slice(&fs::read(&config_path)?)?;
            validate_config(&config, &model_name)?;
            let key = ModelKey {
                name: model_name.clone(),
                version,
            };
            models.insert(
                key.clone(),
                LoadedModel {
                    key,
                    config,
                    directory: version_entry,
                },
            );
        }
    }
    if models.is_empty() {
        return Err(Error::new(
            ErrorCode::Unavailable,
            "model repository contains no loadable versions",
        ));
    }
    Ok(models)
}

fn sorted_directories(root: &Path) -> Result<Vec<PathBuf>> {
    let mut directories = fs::read_dir(root)?
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            entry
                .file_type()
                .ok()
                .filter(std::fs::FileType::is_dir)
                .map(|_| entry.path())
        })
        .collect::<Vec<_>>();
    directories.sort();
    Ok(directories)
}

fn validate_config(config: &ModelConfig, directory_name: &str) -> Result<()> {
    if config.name != directory_name {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            format!(
                "model config name {} does not match directory {directory_name}",
                config.name
            ),
        ));
    }
    if config.max_batch_size == 0
        || config.max_batch_size > 1024
        || config.queue_capacity == 0
        || config.queue_capacity > 1_000_000
    {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            format!("model {} has invalid batching bounds", config.name),
        ));
    }
    Ok(())
}

fn compare_versions(left: &str, right: &str) -> std::cmp::Ordering {
    match (left.parse::<u64>(), right.parse::<u64>()) {
        (Ok(left), Ok(right)) => left.cmp(&right),
        _ => left.cmp(right),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{ModelConfig, ModelRepository};
    use crate::protocol::TensorMetadata;

    fn test_directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "infermesh-repository-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_version(root: &std::path::Path, version: &str) {
        let directory = root.join("model").join(version);
        fs::create_dir_all(&directory).unwrap();
        let tensor = TensorMetadata {
            name: "X".to_owned(),
            datatype: "FP64".to_owned(),
            shape: vec![-1, 1],
        };
        let config = ModelConfig {
            name: "model".to_owned(),
            platform: "cpu".to_owned(),
            backend: "deterministic_cpu".to_owned(),
            max_batch_size: 8,
            max_queue_delay_us: 1000,
            queue_capacity: 16,
            default_timeout_us: 100_000,
            llm: false,
            cache_blocks: 16,
            cache_block_tokens: 4,
            inputs: vec![tensor.clone()],
            outputs: vec![tensor],
        };
        fs::write(
            directory.join("model.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn resolves_latest_numeric_version_and_reloads_atomically() {
        let root = test_directory();
        write_version(&root, "2");
        write_version(&root, "10");
        let repository = ModelRepository::open(&root).unwrap();
        assert_eq!(repository.resolve("model", None).unwrap().key.version, "10");
        assert_eq!(
            repository.resolve("model", Some("2")).unwrap().key.version,
            "2"
        );

        write_version(&root, "11");
        assert_eq!(repository.reload().unwrap(), 2);
        assert_eq!(repository.resolve("model", None).unwrap().key.version, "11");
        fs::remove_dir_all(root).unwrap();
    }
}
