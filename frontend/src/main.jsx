import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true };
  const paths = {
    plus: <><path d="M12 5v14"/><path d="M5 12h14"/></>,
    menu: <><path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h16"/></>,
    upload: <><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 20h14"/></>,
    sun: <circle cx="12" cy="12" r="4"/>,
    moon: <path d="M20 15.2A7.5 7.5 0 0 1 8.8 4 7 7 0 1 0 20 15.2Z"/>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/></>,
    chevron: <path d="m7 10 5 5 5-5"/>,
    send: <><path d="m4 4 16 8-16 8 3-8-3-8Z"/><path d="M7 12h13"/></>,
    paperclip: <path d="m21.4 11.6-8.7 8.7a6 6 0 0 1-8.5-8.5l9.4-9.4a4 4 0 0 1 5.7 5.7L10 17.4a2 2 0 0 1-2.8-2.8l8.5-8.5"/>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-1.4 1.4-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.1h-2v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1-1.4-1.4.1-.1A1.7 1.7 0 0 0 8.6 15a1.7 1.7 0 0 0-1.6-1h-.1v-2H7a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1 1.4-1.4.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.6v-.1h2v.1a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1 1.4 1.4-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.1v2h-.1a1.7 1.7 0 0 0-1.6 1Z"/></>,
    search: <><circle cx="11" cy="11" r="6"/><path d="m16 16 4 4"/></>,
    close: <><path d="m6 6 12 12"/><path d="M18 6 6 18"/></>,
  };
  return <svg {...common}>{paths[name]}</svg>;
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('resrag-theme') || 'light');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [document, setDocument] = useState(null);
  const [docStatus, setDocStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [sources, setSources] = useState([]);
  const [provider, setProvider] = useState(localStorage.getItem('resrag-provider') || 'groq');
  const [model, setModel] = useState(localStorage.getItem('resrag-model') || 'openai/gpt-oss-20b');
  const [providers, setProviders] = useState([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const fileRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('resrag-theme', theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem('resrag-provider', provider);
    localStorage.setItem('resrag-model', model);
  }, [provider, model]);

  useEffect(() => {
    fetch(`${API_URL}/api/providers`)
      .then((r) => r.json())
      .then((data) => setProviders(data.providers || []))
      .catch(() => setProviders([]));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

  useEffect(() => {
    if (!document?.document_id || docStatus?.status === 'full') return undefined;
    const timer = setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/api/documents/${document.document_id}`);
        if (!response.ok) return;
        const status = await response.json();
        setDocStatus(status);
      } catch {
        // Keep the fast index usable if the status request fails.
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [document?.document_id, docStatus?.status]);

  const activeProvider = useMemo(() => providers.find((item) => item.id === provider), [providers, provider]);

  const newChat = () => {
    setMessages([]);
    setSources([]);
    setSourcesOpen(false);
    inputRef.current?.focus();
  };

  const upload = async (file) => {
    if (!file || file.type !== 'application/pdf') return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${API_URL}/api/documents`, { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not upload the PDF.');
      setDocument(data);
      setDocStatus({ status: data.status });
      setMessages([]);
      setSources([]);
      setSourcesOpen(false);
    } catch (error) {
      setMessages((current) => [...current, { role: 'assistant', content: `I couldn't open that PDF: ${error.message}` }]);
    } finally {
      setUploading(false);
    }
  };

  const submit = async () => {
    const question = input.trim();
    if (!question || !document || sending) return;
    setInput('');
    const history = messages.slice(-8).map(({ role, content }) => ({ role, content }));
    setMessages((current) => [...current, { role: 'user', content: question }]);
    setSending(true);
    setSources([]);
    setSourcesOpen(false);
    let assistant = '';
    let receivedSources = [];
    const messageIndex = messages.length + 1;
    setMessages((current) => [...current, { role: 'assistant', content: '', streaming: true }]);
    try {
      const response = await fetch(`${API_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ document_id: document.document_id, question, provider, model, history }),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || 'The server could not answer that question.');
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';
        for (const event of events) {
          const lines = event.split('\n');
          const eventType = lines.find((line) => line.startsWith('event:'))?.slice(6).trim() || 'message';
          const dataLine = lines.find((line) => line.startsWith('data:'));
          if (!dataLine) continue;
          const data = JSON.parse(dataLine.slice(5).trim());
          if (eventType === 'sources') {
            receivedSources = data;
            setSources(data);
          } else if (eventType === 'error') {
            throw new Error(data);
          } else if (eventType === 'done') {
            // Stream finished.
          } else {
            assistant += data;
            setMessages((current) => {
              const next = [...current];
              next[messageIndex] = { role: 'assistant', content: assistant, streaming: true };
              return next;
            });
          }
        }
      }
      setMessages((current) => {
        const next = [...current];
        next[messageIndex] = { role: 'assistant', content: assistant || 'I could not generate an answer.', sources: receivedSources };
        return next;
      });
    } catch (error) {
      setMessages((current) => {
        const next = [...current];
        next[messageIndex] = { role: 'assistant', content: `I couldn't answer that: ${error.message}` };
        return next;
      });
    } finally {
      setSending(false);
    }
  };

  const onKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const statusLabel = document ? (docStatus?.status === 'full' ? 'Ready' : 'Preparing search') : 'No document';

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-top">
          <button className="icon-button mobile-toggle" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar"><Icon name="menu" /></button>
          {sidebarOpen && <div className="wordmark">ResRAG</div>}
          {sidebarOpen && <button className="icon-button" onClick={newChat} aria-label="New chat"><Icon name="plus" /></button>}
        </div>

        {sidebarOpen && <button className="new-chat" onClick={newChat}><Icon name="plus" size={17} /> <span>New chat</span></button>}

        {sidebarOpen && (
          <div className="sidebar-section">
            <div className="section-label">Document</div>
            <div
              className={`upload-box ${dragging ? 'dragging' : ''}`}
              onClick={() => fileRef.current?.click()}
              onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
              onDragOver={(e) => e.preventDefault()}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files?.[0]); }}
            >
              <div className="upload-icon"><Icon name={uploading ? 'search' : 'upload'} size={18} /></div>
              <div>
                <div className="upload-title">{uploading ? 'Opening PDF…' : 'Add a PDF'}</div>
                <div className="upload-subtitle">Drop a file here or browse</div>
              </div>
            </div>
            <input ref={fileRef} type="file" accept="application/pdf,.pdf" hidden onChange={(e) => upload(e.target.files?.[0])} />
            {document && (
              <div className="document-card">
                <div className="document-icon"><Icon name="file" size={17} /></div>
                <div className="document-copy">
                  <div className="document-name" title={document.filename}>{document.filename}</div>
                  <div className="document-meta">{document.pages ? `${document.pages} pages` : 'PDF'} · {statusLabel}</div>
                </div>
                <span className={`status-dot ${docStatus?.status === 'full' ? 'ready' : ''}`} />
              </div>
            )}
          </div>
        )}

        {sidebarOpen && <div className="sidebar-spacer" />}

        {sidebarOpen && (
          <div className="sidebar-bottom">
            <div className="settings-card">
              <div className="settings-row"><Icon name="settings" size={16} /><span>Settings</span></div>
              <label className="field-label">Provider</label>
              <div className="select-wrap"><select value={provider} onChange={(e) => setProvider(e.target.value)}>{providers.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select><Icon name="chevron" size={15} /></div>
              <label className="field-label">Model</label>
              <input className="text-field" value={model} onChange={(e) => setModel(e.target.value)} placeholder={activeProvider?.model_env || 'Model ID'} />
            </div>
            <button className="theme-row" onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')}>
              <span className="theme-left"><Icon name={theme === 'dark' ? 'moon' : 'sun'} size={17} /><span>{theme === 'dark' ? 'Dark mode' : 'Light mode'}</span></span>
              <span className="theme-value">{theme === 'dark' ? 'On' : 'On'}</span>
            </button>
          </div>
        )}
      </aside>

      <main className="main-pane">
        <header className="topbar">
          <div className="topbar-left">
            {!sidebarOpen && <button className="icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar"><Icon name="menu" /></button>}
            <div className="topbar-title">{document ? document.filename : 'New chat'}</div>
          </div>
          <div className="topbar-right">
            <button className="icon-button mobile-theme" onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme"><Icon name={theme === 'dark' ? 'moon' : 'sun'} size={17} /></button>
          </div>
        </header>

        <section className={`conversation ${messages.length === 0 ? 'empty-conversation' : ''}`}>
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-mark">R</div>
              <h1>{document ? 'Ask anything about this PDF' : 'What can I help you find?'}</h1>
              <p>{document ? 'Answers are grounded in the document, with page references to the original source.' : 'Upload a PDF to start a grounded conversation with your document.'}</p>
              {!document && <div className="quick-actions"><button onClick={() => fileRef.current?.click()}> <Icon name="upload" size={16} /> Upload PDF</button><button onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')}><Icon name={theme === 'dark' ? 'moon' : 'sun'} size={16} /> {theme === 'dark' ? 'Light' : 'Dark'} mode</button></div>}
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <article key={`${index}-${message.role}`} className={`message ${message.role}`}>
                  {message.role === 'user' ? (
                    <div className="user-bubble">{message.content}</div>
                  ) : (
                    <div className="assistant-block">
                      <div className="assistant-avatar">R</div>
                      <div className="assistant-body">
                        <div className="assistant-name">ResRAG</div>
                        <div className="assistant-text">{message.content}{message.streaming && <span className="cursor" />}</div>
                        {message.sources?.length > 0 && <button className="source-toggle" onClick={() => { setSources(message.sources); setSourcesOpen((current) => !current); }}>{message.sources.length} sources <Icon name="chevron" size={14} /></button>}
                      </div>
                    </div>
                  )}
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        <div className="composer-wrap">
          {sources.length > 0 && sourcesOpen && (
            <div className="source-panel">
              <div className="source-header"><span>Sources</span><button className="icon-button" onClick={() => setSourcesOpen(false)}><Icon name="close" size={16} /></button></div>
              {sources.map((source, index) => (
                <div className="source-item" key={`${source.page}-${index}`}>
                  <div className="source-top"><span>Page {source.page}</span>{source.section && <span>{source.section}</span>}</div>
                  <div className="source-content">{source.text}</div>
                </div>
              ))}
            </div>
          )}
          <div className="composer">
            <button className="composer-icon" onClick={() => fileRef.current?.click()} aria-label="Attach PDF"><Icon name="paperclip" size={18} /></button>
            <textarea ref={inputRef} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={onKeyDown} placeholder={document ? 'Message ResRAG…' : 'Upload a PDF to start…'} disabled={!document || sending} rows={1} />
            <button className={`send-button ${input.trim() && document ? 'active' : ''}`} onClick={submit} disabled={!input.trim() || !document || sending} aria-label="Send"><Icon name="send" size={17} /></button>
          </div>
          <div className="composer-note">ResRAG can make mistakes. Check important information against the original document.</div>
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
