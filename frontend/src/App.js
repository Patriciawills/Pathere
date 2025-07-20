import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Main Dashboard Component
const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/stats`);
      setStats(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching stats:", error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-500 mx-auto"></div>
          <p className="mt-4 text-indigo-600 font-medium">Initializing Learning Engine...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-lg">🧠</span>
                </div>
              </div>
              <div className="ml-4">
                <h1 className="text-2xl font-bold text-gray-900">Grammar & Vocabulary Engine</h1>
                <p className="text-sm text-gray-500">Human-like Language Learning System</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Active
              </span>
              <span className="text-sm text-gray-500">v{stats?.system?.version || '1.0.0'}</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {/* Stats Overview */}
        <div className="mb-8">
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            <StatsCard 
              title="PDF Files" 
              value={stats?.database?.pdf_files || 0}
              icon="📚"
              color="bg-blue-500"
            />
            <StatsCard 
              title="Language Data" 
              value={stats?.database?.language_data || 0}
              icon="💬"
              color="bg-green-500"
            />
            <StatsCard 
              title="Queries Processed" 
              value={stats?.database?.queries || 0}
              icon="🔍"
              color="bg-purple-500"
            />
            <StatsCard 
              title="Learning Feedback" 
              value={stats?.database?.feedback || 0}
              icon="🎯"
              color="bg-orange-500"
            />
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PDFUploadSection />
          <QuerySection />
        </div>

        <div className="mt-6">
          <DataVisualization stats={stats} />
        </div>
      </div>
    </div>
  );
};

// Stats Card Component
const StatsCard = ({ title, value, icon, color }) => (
  <div className="bg-white overflow-hidden shadow rounded-lg">
    <div className="p-5">
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <div className={`h-8 w-8 ${color} rounded-md flex items-center justify-center text-white text-lg`}>
            {icon}
          </div>
        </div>
        <div className="ml-5 w-0 flex-1">
          <dl>
            <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
            <dd className="text-lg font-medium text-gray-900">{value.toLocaleString()}</dd>
          </dl>
        </div>
      </div>
    </div>
  </div>
);

// PDF Upload Section Component
const PDFUploadSection = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [processing, setProcessing] = useState(false);

  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setUploading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API}/upload-pdf`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setUploadResult(response.data);
      setUploading(false);
    } catch (error) {
      console.error('Upload error:', error);
      setUploading(false);
    }
  };

  const processPDF = async (processingType) => {
    if (!uploadResult) return;

    setProcessing(true);
    try {
      const response = await axios.post(`${API}/process-pdf`, {
        pdf_file_id: uploadResult.file_id,
        processing_type: processingType
      });
      
      // Show success message or update UI
      console.log('Processing result:', response.data);
      alert(`Successfully processed as ${processingType}! Extracted ${response.data.data?.entries?.length || 'some'} entries.`);
      setProcessing(false);
    } catch (error) {
      console.error('Processing error:', error);
      alert('Error processing PDF. Please try again.');
      setProcessing(false);
    }
  };

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
        📖 Dictionary & Grammar Book Upload
      </h3>
      
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
        {!file ? (
          <>
            <div className="text-gray-400 mb-2">
              <svg className="mx-auto h-12 w-12" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <p className="text-sm text-gray-600">Upload Oxford Dictionary or Grammar Book PDF</p>
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              className="mt-2 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
              disabled={uploading}
            />
          </>
        ) : (
          <div>
            <p className="text-sm font-medium text-gray-900">✅ {file.name}</p>
            {uploading && <p className="text-sm text-gray-500 mt-2">Uploading...</p>}
            {uploadResult && (
              <div className="mt-4 space-y-2">
                <p className="text-sm text-green-600">Upload successful!</p>
                <div className="flex space-x-2 justify-center">
                  <button
                    onClick={() => processPDF('dictionary')}
                    disabled={processing}
                    className="px-4 py-2 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 disabled:opacity-50"
                  >
                    {processing ? 'Processing...' : 'Process as Dictionary'}
                  </button>
                  <button
                    onClick={() => processPDF('grammar')}
                    disabled={processing}
                    className="px-4 py-2 bg-green-500 text-white text-sm rounded hover:bg-green-600 disabled:opacity-50"
                  >
                    {processing ? 'Processing...' : 'Process as Grammar'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Query Section Component
const QuerySection = () => {
  const [query, setQuery] = useState('');
  const [queryType, setQueryType] = useState('meaning');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post(`${API}/query`, {
        query_text: query,
        language: 'english',
        query_type: queryType
      });
      setResult(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Query error:', error);
      setResult({ error: 'Failed to process query' });
      setLoading(false);
    }
  };

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
        🔍 Query Language Engine
      </h3>
      
      <div className="space-y-4">
        <div>
          <select
            value={queryType}
            onChange={(e) => setQueryType(e.target.value)}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="meaning">Word Meaning</option>
            <option value="grammar">Grammar Rules</option>
            <option value="usage">Usage Examples</option>
          </select>
        </div>
        
        <div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter word or phrase..."
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
          />
        </div>
        
        <button
          onClick={handleQuery}
          disabled={loading || !query.trim()}
          className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Query Engine'}
        </button>
        
        {result && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            {result.error ? (
              <p className="text-red-600">{result.error}</p>
            ) : (
              <div>
                <div className="flex justify-between items-center mb-2">
                  <h4 className="font-medium text-gray-900">Result:</h4>
                  <span className="text-xs text-gray-500">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="text-gray-700">
                  {result.result?.definition && (
                    <p><strong>Definition:</strong> {result.result.definition}</p>
                  )}
                  {result.result?.examples && (
                    <div className="mt-2">
                      <strong>Examples:</strong>
                      <ul className="list-disc list-inside ml-2">
                        {result.result.examples.map((example, idx) => (
                          <li key={idx} className="text-sm">{example}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {result.context?.related_words && (
                    <div className="mt-2">
                      <strong>Related:</strong>
                      <span className="ml-1 text-sm">
                        {result.context.related_words.join(', ')}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Data Visualization Component
const DataVisualization = ({ stats }) => (
  <div className="bg-white shadow rounded-lg p-6">
    <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
      📊 System Overview
    </h3>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <h4 className="font-medium text-gray-700 mb-2">Learning Engine Status</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span>Memory Usage:</span>
            <span className="font-mono">{stats?.learning_engine?.memory_usage || 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span>Rules Learned:</span>
            <span className="font-mono">{stats?.learning_engine?.rules_count || 0}</span>
          </div>
          <div className="flex justify-between">
            <span>Vocabulary Size:</span>
            <span className="font-mono">{stats?.learning_engine?.vocabulary_size || 0}</span>
          </div>
        </div>
      </div>
      <div>
        <h4 className="font-medium text-gray-700 mb-2">Knowledge Graph</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span>Total Nodes:</span>
            <span className="font-mono">{stats?.knowledge_graph?.nodes_count || 0}</span>
          </div>
          <div className="flex justify-between">
            <span>Relationships:</span>
            <span className="font-mono">{stats?.knowledge_graph?.edges_count || 0}</span>
          </div>
          <div className="flex justify-between">
            <span>Languages:</span>
            <span className="font-mono">{stats?.system?.active_languages?.join(', ') || 'English'}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
);

function App() {
  return (
    <div className="App">
      <Dashboard />
    </div>
  );
}

export default App;