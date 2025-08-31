import "./Sidebar.css";
import { useContext, useEffect } from "react";
import { MyContext } from "./MyContext.jsx";
import { v1 as uuidv1 } from "uuid";

function Sidebar() {
  const {
    allThreads,
    currThreadId,
    setNewChat,
    setPrompt,
    setReply,
    setCurrThreadId,
    setPrevChats,
    sidebarCollapsed,
    fetchThreads,
    fetchMessages,
    deleteThread
  } = useContext(MyContext);

  useEffect(() => {
    fetchThreads();
  }, [currThreadId]);

  const createNewChat = () => {
    setNewChat(true);
    setPrompt("");
    setReply(null);
    setCurrThreadId(uuidv1());
    setPrevChats([]);
  };

  const changeThread = async (newThreadId) => {
    await fetchMessages(newThreadId);
  };

  const truncateTitle = (title, maxLength = 25) => {
    return title.length > maxLength ? title.substring(0, maxLength) + "..." : title;
  };

  return (
    <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
      {/* Header with New Chat Button */}
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={createNewChat}>
          <div className="logo">✨</div>
          <span>New Chat</span>
        </button>
      </div>

      {/* History */}
      <div className="history">
        <div className="history-title">Recent Chats</div>
        {allThreads?.length > 0 ? (
          <ul>
            {allThreads.map((thread, idx) => (
              <li
                key={idx}
                className={thread.threadId === currThreadId ? 'active' : ''}
                onClick={() => changeThread(thread.threadId)}
                title={thread.title}
              >
                <span className="thread-title">
                  {truncateTitle(thread.title)}
                </span>
                <i
                  className="fas fa-trash"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteThread(thread.threadId);
                  }}
                  title="Delete thread"
                ></i>
              </li>
            ))}
          </ul>
        ) : (
          <div className="history-empty">
            No chat history yet.<br />
            Start a conversation to see your chats here.
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
      
      </div>
    </div>
  );
}

export default Sidebar;
