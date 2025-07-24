import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await axios.post('http://localhost:5000/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    setResult(res.data.result);
  };

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      height: '100vh',
      width: '100%',
      background: 'linear-gradient(135deg,rgb(18, 86, 56), #1e3c72, #2a5298)',
    }}>
      <h1 style={{paddingTop: '20px', paddingBottom: '10px',textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)',}}>Your Personalized AI Career Coach</h1>
      <p style={{width: '70%', fontFamily: 'sans-serif', fontStyle: 'italic', paddingBottom: '10px'}}>Choosing the right career path can be overwhelming for students, especially with rapidly evolving job markets and 
        a sea of options. Our AI-powered career assistant helps make smarter, data-driven decisions by analyzing your 
        academic background, interests, and skills. Simply upload your resume, transcript, or CV (PDF or DOCX), and our 
        system builds a structured profile of you. Using advanced AI techniques like semantic search and language models, 
        we match your profile with real-world data from job listings, salary trends, and course catalogs. In just seconds, 
        you'll receive personalized career suggestions, growth opportunities, and learning paths — all tailored to your unique 
        strengths and aspirations.</p>
      
      <input type="file" accept=".pdf,.docx" onChange={handleFileChange} style={{borderRadius: '5px'}}/>
      <button onClick={handleSubmit} disabled={!file} style={{ marginLeft: '10px', borderRadius: '5px'}}>
        Analyze
      </button>
      <pre style={{ height: '300px', width: '1000px', marginTop: '2rem', background: '#eee', padding: '1rem', borderRadius: '5px'}}>
        {result}
      </pre>
    </div>
  );
}

export default App;
