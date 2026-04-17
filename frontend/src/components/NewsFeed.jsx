import React from "react";

/**
 * NewsFeed – Displays news articles with sentiment labels and event tags.
 * Data source: /api/news/<asset> → news_pipeline.py → MongoDB news collection
 */
export default function NewsFeed({ news, loading }) {
  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <span>Loading news...</span>
      </div>
    );
  }

  if (!news || news.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📰</div>
        <p>No news articles available</p>
      </div>
    );
  }

  const formatTime = (ts) => {
    if (!ts) return "";
    const d = new Date(ts);
    const now = new Date();
    const diff = now - d;
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (hours < 1) return "Just now";
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return d.toLocaleDateString("en-IN", { month: "short", day: "numeric", year: "numeric" });
  };

  const getSentimentClass = (label) => {
    if (!label) return "neutral";
    return label.toLowerCase();
  };

  return (
    <ul className="news-list">
      {news.map((item, idx) => (
        <li className="news-item" key={idx}>
          <div className="news-headline">
            {item.link ? (
              <a
                href={item.link}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: "inherit", textDecoration: "none" }}
              >
                {item.headline}
              </a>
            ) : (
              item.headline
            )}
          </div>
          <div className="news-meta">
            <span>{item.source}</span>
            <span>•</span>
            <span>{formatTime(item.timestamp)}</span>
            {item.sentiment_label && (
              <span className={`news-sentiment ${getSentimentClass(item.sentiment_label)}`}>
                {item.sentiment_label}
                {item.sentiment_score != null && ` (${Number(item.sentiment_score).toFixed(2)})`}
              </span>
            )}
            {item.event_type && item.event_type !== "Other" && (
              <span className="news-tag">{item.event_type}</span>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
