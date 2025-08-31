import "./ChatWindow.css";
import Chat from "./Chat.jsx";
import { MyContext } from "./MyContext.jsx";
import { useContext, useState, useEffect } from "react";
import { ScaleLoader } from "react-spinners";
import './LoadingAnimations.css';
import './ScrollAnimations.css';

function ChatWindow() {
  const {
    prompt,
    setPrompt,
    reply,
    setReply,
    currThreadId,
    prevChats,
    setPrevChats,
    setNewChat,
    sidebarCollapsed,
    setSidebarCollapsed,
    darkMode,
    toggleTheme
  } = useContext(MyContext);
  
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [mlFlags, setMlFlags] = useState(null);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [showMlPanel, setShowMlPanel] = useState(true);
  const [mlCollapsed, setMlCollapsed] = useState(false);

  // Scroll to top functionality
  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 300);
      
      // Update scroll progress
      const scrollProgress = document.querySelector('.scroll-progress');
      if (scrollProgress) {
        const scrollPercent = (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100;
        scrollProgress.style.width = `${scrollPercent}%`;
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  const getReply = async () => {
    if (!prompt.trim()) return;
  setMlFlags(null);
    setLoading(true);
    setNewChat(false);
    
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ prompt })
    };

    try {
      // Use relative path so Vite dev server proxy forwards to backend (or set VITE_API_BASE_URL in production)
      const response = await fetch(`/api/thread`, options);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const res = await response.json();
      console.log(res);
      // backend returns openai_output and risk_prediction
      const text = res.openai_output || res.reply || 'No reply';
      setReply(text);
      // show ML panel (placeholder) each time a new assistant output is produced
      setShowMlPanel(true);
      // if backend included ML flags, show them immediately
      if (res.risk_prediction || res.prediction || res.ml) {
        const prediction = res.risk_prediction || res.prediction || (res.ml && res.ml.prediction);
        const confidence = res.risk_confidence || res.confidence || (res.ml && res.ml.confidence) || 0;
        const probabilities = res.probabilities || (res.ml && res.ml.probabilities) || null;
        setMlFlags({ prediction, confidence, probabilities, note: res.note });
      }
    } catch(err) {
      console.error('Fetch error:', err);
      setReply("Sorry, I encountered an error. Please try again.");
    }
    
    setLoading(false);
  };

  // Streaming variant: call /api/thread/stream and process SSE-like chunks
  const getReplyStream = async () => {
    if (!prompt.trim()) return;
    setMlFlags(null);
    setLoading(true);
    setNewChat(false);

    try {
      const res = await fetch('/api/thread/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let assembledText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const raw = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 2);
          if (!raw) continue;
          const line = raw.replace(/^data:\s*/, '');
          try {
            const payload = JSON.parse(line);
            if (payload.type === 'openai') {
              assembledText = payload.text;
              setReply(assembledText);
              // show ML panel (placeholder) when assistant text appears
              setShowMlPanel(true);
            } else if (payload.type === 'ml') {
              setMlFlags({ prediction: payload.prediction, confidence: payload.confidence, probabilities: payload.probabilities });
              // ensure panel is visible when ML flags arrive
              setShowMlPanel(true);
            } else if (payload.type === 'error') {
              setReply('Error: ' + payload.message);
            }
          } catch (e) {
            console.error('stream parse error', e, line);
          }
        }
      }

    } catch (err) {
      console.error('Stream fetch error', err);
      setReply('Sorry, streaming failed.');
    }

    setLoading(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!loading && prompt.trim()) {
  if (streaming) getReplyStream();
  else getReply();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Append new chat to prevChats
  useEffect(() => {
    if (prompt && reply) {
      setPrevChats(prevChats => ([
        ...prevChats,
        { role: "user", content: prompt },
        { role: "assistant", content: reply }
      ]));
    }
    setPrompt("");
  }, [reply]);

  return (
    <div className={`chatWindow ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      {/* Scroll Progress Indicator */}
      <div className="scroll-progress"></div>

      {/* Navbar */}
      <div className="navbar">
        <div className="navbar-left">
          <button 
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            title="Toggle sidebar"
          >
            <i className="fas fa-bars"></i>
          </button>
          <span>स्पष्टGPT</span>
        </div>
        <div className="navbar-right">
          <button 
            className="theme-toggle"
            onClick={toggleTheme}
            title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
          >
            <i className={`fas ${darkMode ? 'fa-sun' : 'fa-moon'}`}></i>
          </button>
          <div className="userIconDiv">
            <div className="userIcon">
              <i className="fas fa-user" style={{fontSize: '14px'}}></i>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Area */}
      {loading ? (
        <div className="loading-container">
          {/* Neural Network Loader */}
          <div className="neural-loader">
            <div className="neural-node"></div>
            <div className="neural-node"></div>
            <div className="neural-node"></div>
            <div className="neural-node"></div>
            <div className="neural-node"></div>
            <div className="neural-node"></div>
          </div>
          
          {/* Alternative loaders - uncomment to use */}
          {/* 
          <div className="quantum-loader">
            <div className="quantum-dot"></div>
            <div className="quantum-dot"></div>
            <div className="quantum-dot"></div>
          </div>
          */}
          
          {/* 
          <div className="holographic-loader">
            <div className="holographic-ring"></div>
            <div className="holographic-ring"></div>
            <div className="holographic-ring"></div>
            <div className="holographic-core"></div>
          </div>
          */}
          
          {/* 
          <div className="matrix-loader">
            <div className="matrix-column"></div>
            <div className="matrix-column"></div>
            <div className="matrix-column"></div>
            <div className="matrix-column"></div>
            <div className="matrix-column"></div>
            <div className="matrix-column"></div>
          </div>
          */}
        </div>
      ) : (
        <Chat />
      )}

      {/* Scroll to Top Button */}
      <button 
        className={`scroll-to-top ${showScrollTop ? 'visible' : ''}`}
        onClick={scrollToTop}
        title="Scroll to top"
      >
        <i className="fas fa-arrow-up"></i>
      </button>

      {/* Chat Input */}
      <div className="chatInput">
        <form onSubmit={handleSubmit} className="inputBox">
          <label htmlFor="chat-input" className="sr-only">Message input</label>
          <input
            value={prompt}
            id="chat-input"
            aria-label="Type your message"
            onChange={(e) => setPrompt(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message here..."
            disabled={loading}
          />
          <div className="controls">
            <label className="control-label" htmlFor="stream-toggle">Stream</label>
            <button
              type="button"
              id="stream-toggle"
              className={"stream-toggle" + (streaming ? ' active' : '')}
              onClick={() => setStreaming(s => !s)}
              aria-pressed={streaming}
              title="Toggle streaming (OpenAI output appears immediately; ML flags arrive when ready)">
              {streaming ? 'ON' : 'OFF'}
            </button>
            <button
              id="submit"
              type="submit"
              disabled={loading || !prompt.trim()}
              style={{
                opacity: loading || !prompt.trim() ? 0.5 : 1,
                cursor: loading || !prompt.trim() ? 'not-allowed' : 'pointer'
              }}
            >
              <i className="fas fa-paper-plane"></i>
            </button>
          </div>
        </form>
        
      </div>
      {/* ML Flags panel */}
      {showMlPanel && (
        <div className={"ml-flags-panel" + (mlCollapsed ? ' collapsed' : '')} role="status" aria-live="polite">
          {mlCollapsed ? (
            <div className="collapsed-icon" title="Expand ML flags" onClick={() => setMlCollapsed(false)}>⚑</div>
          ) : (
            <>
              <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                <div className="panel-row">
                  <strong style={{fontSize:14}}>ML Flags</strong>
                </div>
                <div style={{display:'flex', gap:8}}>
                  <button type="button" className="stream-toggle" onClick={() => setMlCollapsed(true)} title="Collapse">‒</button>
                  <button type="button" className="stream-toggle" onClick={() => setShowMlPanel(false)} title="Hide">✕</button>
                </div>
              </div>
              <div style={{marginTop:10}}>
                {mlFlags ? (
                  <div>
                    {mlFlags.prediction === 'hallucination' ? (
                      <div className="hallucination-warning">
                        <strong>Hallucination detected</strong>
                        <div>Confidence: {mlFlags.confidence.toFixed(3)}</div>
                        <div style={{marginTop:8}}>{mlFlags.note || 'The model flagged this output as potentially misleading.'}</div>
                      </div>
                    ) : (
                      <div>
                        <div className="flag-row">
                          {mlFlags.probabilities ? (
                            Object.entries(mlFlags.probabilities)
                              .sort((a,b)=>b[1]-a[1])
                              .map(([k,v],i)=> (
                                <span key={k} className={"flag-badge flag-"+k + (k===mlFlags.prediction? ' flag-main':'') } title={`${k}: ${(v*100).toFixed(1)}%`}>
                                  {k} · {(v*100).toFixed(1)}%
                                </span>
                              ))
                          ) : (
                            <span className={"flag-badge flag-" + (mlFlags.prediction || 'bias') + ' flag-main'}>{mlFlags.prediction} · {(mlFlags.confidence*100).toFixed(1)}%</span>
                          )}
                        </div>
                        <div style={{marginTop:8,fontSize:12,opacity:0.9}}>Confidence: {(mlFlags.confidence * 100).toFixed(1)}%</div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="ml-empty">ML flags will appear here when available</div>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default ChatWindow;
