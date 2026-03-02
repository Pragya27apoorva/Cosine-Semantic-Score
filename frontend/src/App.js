import React, { useState } from "react";
import axios from "axios";
import ResultChart from "./components/ResultChart";
import "./App.css";

function App() {
  const [type, setType] = useState("text");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append("input_type", type);
    formData.append("text", text);

    const res = await axios.post("http://localhost:8000/query", formData);
    setResult(res.data);
  };

  return (
    <div className="container">
      <h1>Cosine Similarity Semantic Explorer</h1>

      <select onChange={(e)=>setType(e.target.value)}>
        <option value="text">Text</option>
        <option value="image">Image</option>
      </select>

      {type === "text" && (
        <textarea
          rows="4"
          placeholder="Enter your query..."
          value={text}
          onChange={(e)=>setText(e.target.value)}
        />
      )}

      <button onClick={handleSubmit}>Analyze</button>

      {result && (
        <div className="result">
          <h2>Average Score: {result.average_score.toFixed(2)}</h2>
          <p>{result.explanation}</p>
          <ResultChart scores={result.cosine_scores} />
        </div>
      )}
    </div>
  );
}

export default App;
