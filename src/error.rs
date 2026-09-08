use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    InvalidArgument,
    NotFound,
    MethodNotAllowed,
    Unavailable,
    ResourceExhausted,
    DeadlineExceeded,
    Internal,
}

impl ErrorCode {
    /// HTTP status this control-plane error code is surfaced as.
    ///
    /// Kept here rather than in the HTTP layer so every transport reports the
    /// same status for the same failure.
    #[must_use]
    pub const fn http_status(self) -> u16 {
        match self {
            Self::InvalidArgument => 400,
            Self::NotFound => 404,
            Self::MethodNotAllowed => 405,
            Self::ResourceExhausted => 429,
            Self::DeadlineExceeded => 408,
            Self::Unavailable => 503,
            Self::Internal => 500,
        }
    }
}

#[derive(Clone, Debug, Error)]
#[error("{code:?}: {message}")]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}

impl Error {
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::new(ErrorCode::Internal, error.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::new(ErrorCode::InvalidArgument, error.to_string())
    }
}
