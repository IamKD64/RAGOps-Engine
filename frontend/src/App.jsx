
import { useEffect, useState } from 'react'
import './App.css'

const MODES = [
  { value: 'vector', label: 'Vector Search' },
  { value: 'bm25', label: 'BM25 Search' },
  { value: 'hybrid', label: 'Hybrid Search' },
  { value: 'hybrid_rerank', label: 'Hybrid + Reranking' },
]

const suggestions = [
  'What is greedy search in text generation?',
  'What is parameter-efficient fine-tuning?',
  'Explain the A2A protocol.',
]

function App() {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState('hybrid_rerank')
  const [k, setK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [expandedSource, setExpandedSource] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')

  // Check whether the FastAPI backend is responding
  useEffect(() => {
    fetch('/api/health')
      .then((response) => {
        if (!response.ok) {
          throw new Error('API unavailable')
        }
        return response.json()
      })
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'))
  }, [])

  // Send a question to the RAG backend
  async function askQuestion(value = question) {
    const trimmed = value.trim()

    if (trimmed.length < 3 || loading) return

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: trimmed,
    }

    setMessages((previous) => [...previous, userMessage])
    setQuestion('')
    setError('')
    setLoading(true)

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: trimmed,
          mode,
          k: Number(k),
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Something went wrong.')
      }

      setMessages((previous) => [
        ...previous,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: data.answer,
          sources: data.sources || [],
          latency: data.latency_ms || {},
        },
      ])
    } catch (err) {
      setError(err.message || 'Unable to connect to the API.')
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(event) {
    event.preventDefault()
    askQuestion()
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">R</div>
          <div>
            <h2>RAG Assistant</h2>
            <span>AI Knowledge Workspace</span>
          </div>
        </div>

        <button
          className="new-chat"
          onClick={() => {
            setMessages([])
            setError('')
            setExpandedSource(null)
          }}
        >
          <span>＋</span> New conversation
        </button>

        <div className="sidebar-section">
          <span className="section-label">WORKSPACE</span>
          <div className="sidebar-item active">
            ◈ &nbsp; AI Document Assistant
          </div>
          <div className="sidebar-item">
            ▤ &nbsp; Knowledge Base
          </div>
          <div className="sidebar-item">
            ⌘ &nbsp; Retrieval Settings
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className={`status-dot ${apiStatus}`} />
            <div>
              <strong>
                {apiStatus === 'checking'
                  ? 'Checking API...'
                  : apiStatus === 'online'
                    ? 'API Connected'
                    : 'API Offline'}
              </strong>
              <small>
                {apiStatus === 'online'
                  ? 'Backend responding'
                  : apiStatus === 'offline'
                    ? 'Check FastAPI server'
                    : 'Connecting to backend'}
              </small>
            </div>
          </div>

          <div className="sidebar-footer">
            Production RAG Assistant · v1.0
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <span className="breadcrumb">WORKSPACE / </span>
            <strong>Document Assistant</strong>
          </div>

          <div className="topbar-status">
            <span className={`status-dot ${apiStatus}`} />
            {apiStatus === 'checking'
              ? 'Checking system'
              : apiStatus === 'online'
                ? 'System online'
                : 'System offline'}
          </div>
        </header>

        <section className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">✳</div>

              <div className="eyebrow">
                YOUR AI KNOWLEDGE ASSISTANT
              </div>

              <h1>
                Ask your documents<span>.</span>
              </h1>

              <p>
                Explore your AI knowledge base with intelligent retrieval,
                reranking, and grounded answers backed by document sources.
              </p>

              <div className="suggestion-grid">
                {suggestions.map((item) => (
                  <button
                    className="suggestion"
                    key={item}
                    onClick={() => askQuestion(item)}
                    disabled={loading}
                  >
                    <span>↗</span>
                    {item}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((message) => (
                <article
                  className={`message ${message.role}`}
                  key={message.id}
                >
                  <div className="message-avatar">
                    {message.role === 'user' ? 'Y' : 'R'}
                  </div>

                  <div className="message-body">
                    <div className="message-author">
                      {message.role === 'user'
                        ? 'You'
                        : 'RAG Assistant'}
                    </div>

                    <div className="message-content">
                      {message.content}
                    </div>

                    {message.sources?.length > 0 && (
                      <div className="sources">
                        <div className="sources-heading">
                          <span>▤</span>
                          Retrieved sources ({message.sources.length})
                        </div>

                        {message.sources.map((source, index) => {
                          const sourceKey = `${message.id}-${index}`

                          return (
                            <div
                              className="source-card"
                              key={`${source.chunk_id}-${index}`}
                            >
                              <button
                                className="source-toggle"
                                onClick={() =>
                                  setExpandedSource(
                                    expandedSource === sourceKey
                                      ? null
                                      : sourceKey,
                                  )
                                }
                              >
                                <span className="source-number">
                                  {index + 1}
                                </span>

                                <span className="source-name">
                                  {source.source}
                                  <small>
                                    Chunk {source.chunk_id}
                                  </small>
                                </span>

                                <span className="source-arrow">
                                  {expandedSource === sourceKey
                                    ? '−'
                                    : '+'}
                                </span>
                              </button>

                              {expandedSource === sourceKey && (
                                <div className="source-text">
                                  {source.text}
                                </div>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    )}

                    {message.latency && (
                      <div className="latency">
                        <span>
                          ◷ Retrieval: {message.latency.retrieval} ms
                        </span>
                        <span>
                          ◷ Generation: {message.latency.generation} ms
                        </span>
                      </div>
                    )}
                  </div>
                </article>
              ))}

              {loading && (
                <div className="loading-message">
                  <div className="message-avatar">R</div>
                  <div>
                    <strong>RAG Assistant</strong>
                    <p>
                      Searching documents and generating an answer...
                    </p>
                    <div className="loading-dots">
                      <i />
                      <i />
                      <i />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="error-message">
              ⚠ {error}
            </div>
          )}
        </section>

        <footer className="composer-area">
          <div className="controls">
            <label>
              RETRIEVAL STRATEGY
              <select
                value={mode}
                onChange={(event) => setMode(event.target.value)}
              >
                {MODES.map((item) => (
                  <option value={item.value} key={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              TOP K RESULTS
              <select
                value={k}
                onChange={(event) => setK(Number(event.target.value))}
              >
                {[3, 5, 7, 10].map((value) => (
                  <option value={value} key={value}>
                    {value} results
                  </option>
                ))}
              </select>
            </label>
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about your AI documents..."
              disabled={loading}
              minLength={3}
            />

            <button
              type="submit"
              disabled={loading || question.trim().length < 3}
            >
              {loading ? '...' : '↑'}
            </button>
          </form>

          <div className="composer-note">
            AI-generated answers can make mistakes. Verify important
            information against the sources.
          </div>
        </footer>
      </main>
    </div>
  )
}

export default App
