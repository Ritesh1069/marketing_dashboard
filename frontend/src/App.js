import React, { useState } from 'react';
import ContentEvaluation from './components/ContentEvaluation';
import './App.css';
import Navbar from './components/Navbar/Navbar'
function App() {
  const [prompt, setPrompt] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8080/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to analyze content. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Navbar/>
    <div className="container">
      <h1 className="title">Content Generation & Analysis</h1>
      
      <form onSubmit={handleSubmit} className="form">
        <textarea
          className="input"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your content prompt here..."
          rows="4"
          required
        />
        <button 
          type="submit" 
          className="button"
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Analyze Content'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}
      
      {data && <ContentEvaluation data={data} />}
    </div>
    </div>
  );
}

export default App; 