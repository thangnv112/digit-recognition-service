from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_path: str = 'weights/mnist_cnn.onnx'
    log_level: str = 'INFO'
    max_image_size_mb: int = 10
    min_contour_area: int = 50
    confidence_threshold: float = 0.5

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
