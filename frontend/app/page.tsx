'use client';

import { useState } from 'react';

export default function Home() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);


 const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  if (!query) return;

  setIsLoading(true);
  setAnswer('');

  try {
    const response = await fetch('http://localhost:8000/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    setAnswer(data.answer);

  } catch (error) {
    console.error('Error fetching answer:', error);
    const errorMessage = error instanceof Error 
      ? `エラー: ${error.message}` 
      : 'An error occurred while fetching the answer.';
    setAnswer(errorMessage);
  } finally {
    setIsLoading(false);
  }
};


  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="w-full max-w-xl">
        <h1 className="text-3xl font-bold mb-6 text-center">RAG System</h1>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question..."
            className="w-full p-3 border border-gray-300 rounded-lg mb-4 text-black"
            disabled={isLoading}
          />
          <button
            type="submit"
            className="w-full bg-blue-500 text-white p-3 rounded-lg hover:bg-blue-600 disabled:bg-blue-300"
            disabled={isLoading}
          >
            {isLoading ? 'Thinking...' : 'Ask'}
          </button>
        </form>
        {answer && (
          <div className="mt-6 p-4 bg-gray-100 rounded-lg">
            <p className="text-gray-800">{answer}</p>
          </div>
        )}
      </div>
    </main>
  );
}