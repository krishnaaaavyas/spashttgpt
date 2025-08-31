// frontend/src/App.jsx
import './App.css';
import ChatWindow from './ChatWindow.jsx';
import Sidebar from './Sidebar.jsx';
import { MyContext } from './MyContext.jsx';
import { useState, useEffect } from 'react';
import { v1 as uuidv1 } from 'uuid';

// Base URL: empty string in dev so Vite proxy handles /api -> backend,
// or set VITE_API_BASE_URL in .env for deployments.
const BASE = import.meta.env.VITE_API_BASE_URL || '';

// frontend/src/App.jsx
// ... other code
async function callApi(method, path, body = null) {
  try {
    const res = await fetch(`${BASE}${path}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: body ? JSON.stringify(body) : undefined,
    });
    // ... rest of the function remains the same
  } catch (error) {
    // ...
  }
}
// ... rest of the file
function App() {
  const [prompt, setPrompt] = useState('');
  const [reply, setReply] = useState(null);
  const [currThreadId, setCurrThreadId] = useState(uuidv1());
  const [prevChats, setPrevChats] = useState([]);
  const [newChat, setNewChat] = useState(true);
  const [allThreads, setAllThreads] = useState([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    return savedTheme ? savedTheme === 'dark' : true;
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const toggleTheme = () => setDarkMode((prev) => !prev);

  // ---- API calls ----
  const fetchThreads = async () => {
    const data = await callApi('GET', '/api/thread');
    if (data) setAllThreads(data);
  };

  const fetchMessages = async (threadId) => {
    setNewChat(false);
    setCurrThreadId(threadId);
    const data = await callApi('GET', `/api/thread/${threadId}`);
    if (data) setPrevChats(data);
  };

  const sendMessage = async () => {
    if (!prompt.trim()) return;

    const userMessage = { role: 'user', content: prompt };
    setPrevChats((prev) => [...prev, userMessage]);

    const data = await callApi('POST', '/api/chat', { threadId: currThreadId, message: prompt });
    if (data) {
      setReply(data.reply);
      setPrevChats((prev) => [...prev, { role: 'assistant', content: data.reply }]);
      if (newChat) {
        await fetchThreads();
        setNewChat(false);
      }
    }
    setPrompt('');
  };

  const deleteThread = async (threadId) => {
    const data = await callApi('DELETE', `/api/thread/${threadId}`);
    if (data) {
      await fetchThreads();
      if (threadId === currThreadId) {
        setNewChat(true);
        setPrompt("");
        setReply(null);
        setCurrThreadId(uuidv1());
        setPrevChats([]);
      }
    }
  };

  const providerValues = {
    prompt,
    setPrompt,
    reply,
    setReply,
    currThreadId,
    setCurrThreadId,
    newChat,
    setNewChat,
    prevChats,
    setPrevChats,
    allThreads,
    setAllThreads,
    sidebarCollapsed,
    setSidebarCollapsed,
    darkMode,
    setDarkMode,
    toggleTheme,
    fetchThreads,
    fetchMessages,
    sendMessage,
    deleteThread
  };

  useEffect(() => {
    fetchThreads();
  }, []);

  return (
    <MyContext.Provider value={providerValues}>
      <div className="app">
        <Sidebar />
        <ChatWindow />
      </div>
    </MyContext.Provider>
  );
}

export default App;
