import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function Icon({ name, size = 18 }) {
  const common = {
    width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
    strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true,
  };
  const paths = {
    plus: <><path d="M12 5v14"/><path d="M5 12h14"/></>,
    menu: <><path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h16"/></>,
    upload: <><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 20h14"/></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/></>,
    chevron: <path d="m7 10 5 5 5-5"/>,
    send: <><path d="m4 4 16 8-16 8 3-8-3-8Z"/><path d="M7 12h13"/></>,
    paperclip: <path d="m21.4 11.6-8.7 8.7a6 6 0 0 1-8.5-8.5l9.4-9.4a4 4 0 0 1 5.7 5.7L10 17.4a2 2 0 0 1-2.8-2.8l8.5-8.5"/>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-1.4 1.4-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.1h-2v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1-1.4-1.4.1-.1A1.7 1.7 0 0 0 8.6 15a1.7 1.7 0 0 0-1.6-1h-.1v-2H7a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1 1.4-1.4.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.6v-.1h2v.1a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1 1.4 1.4-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.1v2h-.1a1.7 1.7 0 0 0-1.6 1Z"/></>,
    sun: <><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></>,
    moon: <path d="M20 15.2A7.5 7.5 0 0 1 8.8 4 7 7 0 1 0 20 15.2Z"/>,
    close: <><path d="m6 6 12 12"/><path d="M18 6 6 18"/></>,
    copy: <><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></>,
    fileText: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h6"/></>,
  };
  return <svg {...common}>{paths[name]}</svg>;
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('resrag-theme') || 'light');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeDocument, setActiveDocument] = useState(null);
  const [docStatus, setDocStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [sourcePanel, setSourcePanel] = useState(null);
  const [provider, setProvider] = useState(localStorage.getItem('resrag-provider') || 'groq');
  const [model, setModel] = useState(localStorage.getItem('resrag-model') || 'openai/gpt-oss-20b');
  const [providers, setProviders] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const inputRef = useRef(null);
  const fileRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    window.document.documentElement.dataset.theme = theme;
    localStorage.setItem('resrag-theme', theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem('resrag-provider', provider);
    localStorage.setItem('resrag-model', model);
  }, [provider, model]);

  useEffect(() => {
    fetch(`${API_URL}/api/providers`)
      .then((response) => response.json())
      .then((data) => setProviders(data.providers || []))
      .catch(() => setProviders([]));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, sending]);

  useEffect(() => {
    if (!activeDocument?.document_id || docStatus?.status === 'full') return undefined;
    const timer = setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/api/documents/${activeDocument.document_id}`);
        if (!response.ok) return;
        setDocStatus(await response.json());
      } catch {
        // Keep the current fast index available.
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [activeDocument?.document_id, docStatus?.status]);

  const activeProvider = useMemo(
    () => providers.find((item) => item.id === provider),
    [providers, provider],
  );

  const statusLabel = activeDocument
    ? (docStatus?.status === 'full' ? 'Ready' : 'Preparing search')
    : 'No document';

  const newChat = () => {
    setMessages([]);
    setSourcePanel(null);
    setInput('');
    inputRef.current?.focus();
  };

  const upload = async (file) => {
    if (!file || (file.type && file.type !== 'application/pdf' && !file.name?.toLowerCase().endsWith('.pdf'))) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${API_URL}/api/documents`, { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not upload the PDF.');
      setActiveDocument(data);
      setDocStatus({ status: data.status });
      setMessages([]);
      setSourcePanel(null);
      setInput('');
      setTimeout(() => inputRef.current?.focus(), 50);
    } catch (error) {
      setMessages([{ role: 'assistant', content: `I couldn't open that PDF: ${error.message}` }]);
    } finally {
      setUploading(false);
    }
  };

  const submit = async () => {
    const question = input.trim();
    if (!question || !activeDocument || sending) return;
    setInput('');
    const history = messages.slice(-8).map(({ role, content }) => ({ role, content }));
    const assistantIndex = messages.length + 1;
    setMessages((current) => [
      ...current,
      { role: 'user', content: question },
      { role: 'assistant', content: '', streaming: true, sources: [] },
    ]);
    setSending(true);
    setSourcePanel(null);
    let assistant = '';
    let receivedSources = [];

    try {
      const response = await fetch(`${API_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({
          document_id: activeDocument.document_id,
          question,
          provider,
          model,
          history,
        }),
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
            setMessages((current) => {
              const next = [...current];
              if (next[assistantIndex]) next[assistantIndex] = { ...next[assistantIndex], sources: data };
              return next;
            });
          } else if (eventType === 'error') {
            throw new Error(data);
          } else if (eventType !== 'done') {
            assistant += data;
            setMessages((current) => {
              const next = [...current];
              if (next[assistantIndex]) next[assistantIndex] = { ...next[assistantIndex], content: assistant, streaming: true };
              return next;
            });
          }
        }
      }

      setMessages((current) => {
        const next = [...current];
        if (next[assistantIndex]) next[assistantIndex] = { role: 'assistant', content: assistant || 'I could not generate an answer.', sources: receivedSources };
        return next;
      });
    } catch (error) {
      setMessages((current) => {
        const next = [...current];
        if (next[assistantIndex]) next[assistantIndex] = { role: 'assistant', content: `I couldn't answer that: ${error.message}` };
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

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <button className="icon-button" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar"><Icon name="menu" /></button>
          {sidebarOpen && <div className="wordmark">ResRAG</div>}
          {sidebarOpen && <button className="icon-button" onClick={newChat} aria-label="New chat"><Icon name="plus" /></button>}
        </div>

        {sidebarOpen && <button className="new-chat" onClick={newChat}><Icon name="plus" size={16} /><span>New chat</span></button>}

        {sidebarOpen && (
          <div className="sidebar-body">
            <div className="sidebar-label">Document</div>
            <div
              className={`upload-card ${dragging ? 'dragging' : ''}`}
              onClick={() => fileRef.current?.click()}
              onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => { event.preventDefault(); setDragging(false); upload(event.dataTransfer.files?.[0]); }}
            >
              <div className="upload-card-icon"><Icon name="upload" size={16} /></div>
              <div className="upload-copy">
                <div className="upload-title">{uploading ? 'Uploading…' : 'Upload PDF'}</div>
                <div className="upload-subtitle">Drop a file or browse your device</div>
              </div>
            </div>
            <input ref={fileRef} type="file" accept="application/pdf,.pdf" hidden onChange={(event) => upload(event.target.files?.[0])} />

            {activeDocument && (
              <div className="document-card">
                <div className="document-icon"><Icon name="fileText" size={16} /></div>
                <div className="document-copy">
                  <div className="document-name" title={activeDocument.filename}>{activeDocument.filename}</div>
                  <div className="document-meta">{activeDocument.pages ? `${activeDocument.pages} pages` : 'PDF'} · {statusLabel}</div>
                </div>
                <span className={`status-dot ${docStatus?.status === 'full' ? 'ready' : ''}`} />
              </div>
            )}
          </div>
        )}

        <div className="sidebar-spacer" />

        {sidebarOpen && (
          <div className="sidebar-footer">
            <button className="sidebar-row" onClick={() => setSettingsOpen((current) => !current)}>
              <span><Icon name="settings" size={16} /> Settings</span>
              <Icon name="chevron" size={15} />
            </button>
            {settingsOpen && (
              <div className="settings-popover">
                <label>Provider</label>
                <select value={provider} onChange={(event) => setProvider(event.target.value)}>
                  {providers.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
                </select>
                <label>Model</label>
                <input value={model} onChange={(event) => setModel(event.target.value)} placeholder={activeProvider?.model_env || 'Model ID'} />
              </div>
            )}
            <button className="sidebar-row" onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')}>
              <span><Icon name={theme === 'dark' ? 'moon' : 'sun'} size={16} /> {theme === 'dark' ? 'Dark mode' : 'Light mode'}</span>
              <span className="muted-end">{theme === 'dark' ? 'On' : 'On'}</span>
            </button>
          </div>
        )}
      </aside>

      <main className="main-pane">
        <header className="topbar">
          <div className="topbar-left">
            {!sidebarOpen && <button className="icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar"><Icon name="menu" /></button>}
            <button className="model-button" onClick={() => setSettingsOpen((current) => !current)}>
              <span>ResRAG</span><Icon name="chevron" size={14} />
            </button>
          </div>
          <div className="topbar-right">
            {activeDocument && <div className="document-pill"><Icon name="file" size={13} /><span>{activeDocument.filename}</span></div>}
            <button className="icon-button" onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme"><Icon name={theme === 'dark' ? 'moon' : 'sun'} size={16} /></button>
          </div>
        </header>

        <section className={`conversation ${messages.length === 0 ? 'conversation-empty' : ''}`}>
          {messages.length === 0 ? (
            <div className="welcome">
              <h1>{activeDocument ? 'Ask anything about your PDF' : 'What can I help you find?'}</h1>
              <p>{activeDocument ? 'Ask about facts, sections, tables, or anything else in the document.' : 'Upload a PDF and start a conversation with its contents.'}</p>
              {activeDocument && (
                <div className="prompt-chips">
                  {['Summarize this document', 'What are the main points?', 'What are the key findings?'].map((prompt) => (
                    <button key={prompt} onClick={() => { setInput(prompt); inputRef.current?.focus(); }}>{prompt}</button>
                  ))}
                </div>
              )}
              {!activeDocument && <button className="welcome-upload" onClick={() => fileRef.current?.click()}><Icon name="upload" size={16} /> Upload PDF</button>}
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <article className={`message-row ${message.role}`} key={`${message.role}-${index}`}>
                  {message.role === 'user' ? (
                    <div className="user-bubble">{message.content}</div>
                  ) : (
                    <div className="assistant-wrap">
                      <div className="assistant-content">{message.content}<span className={`stream-cursor ${message.streaming ? '' : 'hidden'}`} /></div>
                      {message.sources?.length > 0 && !message.streaming && (
                        <button className="sources-link" onClick={() => setSourcePanel(message.sources)}>
                          {message.sources.length} source{message.sources.length === 1 ? '' : 's'}
                        </button>
                      )}
                    </div>
                  )}
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        <div className="composer-area">
          {sourcePanel?.length > 0 && (
            <div className="source-panel">
              <div className="source-panel-header">
                <span>Sources</span>
                <button className="icon-button" onClick={() => setSourcePanel(null)} aria-label="Close sources"><Icon name="close" size={16} /></button>
              </div>
              {sourcePanel.map((source, index) => (
                <div className="source-item" key={`${source.page}-${index}`}>
                  <div className="source-meta">Page {source.page}{source.section ? ` · ${source.section}` : ''} · {source.kind}</div>
                  <div className="source-text">{source.text}</div>
                </div>
              ))}
            </div>
          )}

          <div className="composer-shell">
            <button className="composer-icon" onClick={() => fileRef.current?.click()} aria-label="Attach PDF"><Icon name="paperclip" size={18} /></button>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onKeyDown}
              disabled={!activeDocument || sending}
              rows={1}
              placeholder={activeDocument ? 'Ask a question about your document…' : 'Upload a PDF to start chatting…'}
            />
            <button className={`send-button ${input.trim() && activeDocument ? 'active' : ''}`} onClick={submit} disabled={!input.trim() || !activeDocument || sending} aria-label="Send message"><Icon name="send" size={17} /></button>
          </div>
          <div className="composer-footer">ResRAG may make mistakes. Check important information against the source.</div>
        </div>
      </main>
    </div>
  );
}

createRoot(window.document.getElementById('root')).render(<App />);
