CREATE TABLE IF NOT EXISTS inspection_results (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    image_format VARCHAR(20) NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    prediction VARCHAR(20) NOT NULL,
    confidence REAL NOT NULL,
    priority VARCHAR(20) NOT NULL,
    priority_reason TEXT NOT NULL,
    review_required BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    review_status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    review_notes TEXT,
);