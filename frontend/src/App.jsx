
import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef(null)

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('')
      return
    }

    const url = URL.createObjectURL(selectedFile)
    setPreviewUrl(url)

    return () => URL.revokeObjectURL(url)
  }, [selectedFile])

  function handleFileChange(event) {
    const file = event.target.files?.[0]
    if (!file) return

    if (!file.type.startsWith('image/')) {
      setError('Please choose a valid image file.')
      setSelectedFile(null)
      event.target.value = ''
      return
    }

    setError('')
    setResult(null)
    setSelectedFile(file)
  }

  function removeImage() {
    setSelectedFile(null)
    setResult(null)
    setError('')

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  async function analyzeImage() {
    if (!selectedFile || loading) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(`${API_URL}/predict_damage`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json().catch(() => ({}))

      if (!response.ok) {
        throw new Error(
          data.detail || `Request failed with status ${response.status}`,
        )
      }

      setResult(data)
    } catch (err) {
      setError(
        `Could not analyze the image: ${err.message}. Check that the backend is running.`,
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="dashboard">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">STRUCTURAL INSPECTION PLATFORM</p>
          <h1>Inspection Dashboard</h1>
          <p className="subtitle">
            Monitor structural damage inspections and review results.
          </p>
        </div>

        <div className="status-badge">
          <span className="status-dot" />
          API Connected
        </div>
      </header>

      <section className="summary-grid">
        <article className="summary-card">
          <div className="card-icon">📋</div>
          <p>Total Inspections</p>
          <h2>—</h2>
          <span>Recorded inspections</span>
        </article>

        <article className="summary-card">
          <div className="card-icon">🕒</div>
          <p>Pending Reviews</p>
          <h2>—</h2>
          <span>Inspections awaiting review</span>
        </article>

        <article className="summary-card">
          <div className="card-icon">🔍</div>
          <p>New Inspection</p>
          <h2>{selectedFile ? '1 image' : 'Upload'}</h2>
          <span>Start a structural assessment</span>
        </article>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Start an Inspection</h2>
          <p>
            Upload a photo of a building, bridge, road, or other structure.
          </p>
        </div>

        <div className="upload-area">
          {previewUrl && selectedFile ? (
            <>
              <img
                src={previewUrl}
                alt="Selected structure"
                className="image-preview"
              />

              <h3>Image ready for inspection</h3>
              <p className="file-name">{selectedFile.name}</p>
              <p>
                File size: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>

              <div className="upload-actions">
                <label htmlFor="image-upload" className="action-button">
                  Choose another image
                </label>

                <button
                  type="button"
                  className="remove-button"
                  onClick={removeImage}
                  disabled={loading}
                >
                  Remove image
                </button>
              </div>

              <button
                type="button"
                className="analyze-button"
                onClick={analyzeImage}
                disabled={loading}
              >
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </button>
            </>
          ) : (
            <>
              <div className="upload-icon">↑</div>
              <h3>Upload a structural image</h3>
              <p>
                Choose an image from your computer to preview it before
                damage classification.
              </p>

              <label htmlFor="image-upload" className="action-button">
                Choose Image
              </label>

              <p className="supported-formats">
                Select an image file from your device.
              </p>
            </>
          )}

          <input
            ref={fileInputRef}
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="file-input"
            disabled={loading}
          />

          {loading && (
            <p role="status">Sending image to the classification service…</p>
          )}

          {error && <p className="error-message">{error}</p>}
        </div>
      </section>


{result && (
  <section className="panel">
    <div className="panel-heading">
      <p className="eyebrow">AI CLASSIFICATION REPORT</p>
      <h2>Analysis Result</h2>
      <p>Review the predicted damage category and model confidence.</p>
    </div>

    <div className="result-card">
      <div className="result-topline">
        <span className="result-status-dot" />
        <span>Analysis completed</span>
        <span className="result-model">
          {result.model_version || 'AI model'}
        </span>
      </div>

      <div className="result-main">
        <div>
          <p className="result-label">Predicted damage</p>
          <h3 className="result-prediction">
            {String(result.prediction || 'Unknown')
              .replaceAll('_', ' ')}
          </h3>
          <p className="result-filename">
            {result.filename || selectedFile?.name || 'Uploaded image'}
          </p>
        </div>

        <div className="result-confidence">
          <span>Confidence</span>
          <strong>
            {Number.isFinite(Number(result.confidence))
              ? `${(Number(result.confidence) * 100).toFixed(2)}%`
              : 'N/A'}
          </strong>
        </div>
      </div>

      <div className="result-progress-track">
        <div
          className="result-progress-fill"
          style={{
            width: `${
              Number.isFinite(Number(result.confidence))
                ? Math.max(0, Math.min(100, Number(result.confidence) * 100))
                : 0
            }%`,
          }}
        />
      </div>

      <div className="result-details">
        <div>
          <span>Image format</span>
          <strong>{result.format || 'Unknown'}</strong>
        </div>
        <div>
          <span>Image dimensions</span>
          <strong>
            {result.width && result.height
              ? `${result.width} × ${result.height}`
              : 'Unknown'}
          </strong>
        </div>
        <div>
          <span>Model version</span>
          <strong>{result.model_version || 'Unknown'}</strong>
        </div>
      </div>

      {result.class_probabilities &&
        Object.keys(result.class_probabilities).length > 0 && (
          <div className="probabilities-section">
            <h3>Class probability breakdown</h3>
            <p className="probabilities-note">
              Scores reported by the model for each class.
            </p>

            {Object.entries(result.class_probabilities)
              .sort((a, b) => Number(b[1]) - Number(a[1]))
              .map(([className, probability]) => {
                const score = Number(probability)
                const validScore = Number.isFinite(score)
                  ? Math.max(0, Math.min(1, score))
                  : 0

                return (
                  <div className="probability-row" key={className}>
                    <div className="probability-heading">
                      <span>{className.replaceAll('_', ' ')}</span>
                      <strong>
                        {Number.isFinite(score)
                          ? `${(score * 100).toFixed(2)}%`
                          : 'N/A'}
                      </strong>
                    </div>

                    <div className="probability-track">
                      <div
                        className="probability-fill"
                        style={{ width: `${validScore * 100}%` }}
                      />
                    </div>
                  </div>
                )
              })}
          </div>
        )}

    </div>
  </section>
)}


      <section className="panel">
        <div className="panel-heading">
          <h2>Recent Inspections</h2>
          <p>Your latest structural assessments will appear here.</p>
        </div>

        <div className="empty-state">
          <span>📁</span>
          <p>Inspection history will be connected next.</p>
        </div>
      </section>

      <footer className="dashboard-footer">
        Structural Damage Assessment · Project Dashboard
      </footer>
    </main>
  )
}

export default App
