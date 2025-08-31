import "./Chat.css";
import React, { useContext, useState, useEffect, useRef } from "react";
import { MyContext } from "./MyContext";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/atom-one-dark.css";
import "highlight.js/styles/github.css";

function Chat() {
  const {newChat, prevChats, reply, darkMode} = useContext(MyContext);
  const [latestReply, setLatestReply] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const chatsEndRef = useRef(null);

  const scrollToBottom = () => {
    chatsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [prevChats, latestReply]);

  // Dynamically import highlight.js theme based on current theme
  useEffect(() => {
    const existingLink = document.getElementById('hljs-theme');
    if (existingLink) {
      existingLink.remove();
    }

    const link = document.createElement('link');
    link.id = 'hljs-theme';
    link.rel = 'stylesheet';
    link.href = darkMode 
      ? 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css'
      : 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
    
    document.head.appendChild(link);
  }, [darkMode]);

  useEffect(() => {
    if(reply === null) {
      setLatestReply(null);
      setIsTyping(false);
      return;
    }
    
    if(!prevChats?.length) return;
    
    setIsTyping(true);
    const content = reply.split(" ");
    let idx = 0;
    
    const interval = setInterval(() => {
      setLatestReply(content.slice(0, idx+1).join(" "));
      idx++;
      if(idx >= content.length) {
        clearInterval(interval);
        setIsTyping(false);
      }
    }, 50);
    
    return () => clearInterval(interval);
  }, [prevChats, reply]);

  return (
    <>
      {newChat && (
        <div className="welcome-container">
          <div className="welcome-title">
            ✨ स्पष्टGPT
          </div>
          <p className="welcome-subtitle">
            Welcome! Ask me anything and I'll help you with detailed, thoughtful responses.
          </p>
        </div>
      )}
      
      <div className="chats">
        {prevChats?.map((chat, idx) => (
          <div key={idx} className={chat.role === "user" ? "userDiv" : "gptDiv"}>
            <div className={chat.role === "user" ? "userMessage" : "gptMessage"}>
              {chat.role === "user" ? (
                chat.content
              ) : (
                <ReactMarkdown 
                  rehypePlugins={[rehypeHighlight]}
                  components={{
                    code({node, inline, className, children, ...props}) {
                      return inline ? (
                        <code className="inline-code" {...props}>
                          {children}
                        </code>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    }
                  }}
                >
                  {idx === prevChats.length - 1 && latestReply ? latestReply : chat.content}
                </ReactMarkdown>
              )}
            </div>
          </div>
        ))}
        
        {isTyping && (
          <div className="gptDiv">
            <div className="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          </div>
        )}
        
        <div ref={chatsEndRef} />
      </div>
    </>
  );
}

export default Chat;
