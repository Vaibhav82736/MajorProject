import { useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:5000";

function App() {
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const [form, setForm] = useState({
    age: "", height: "", weight: "",
    gender: "1", ap_hi: "", ap_lo: "",
    cholesterol: "1", gluc: "1",
    smoke: "0", alco: "0", active: "1"
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  // LOGIN
  const login = async () => {
    const res = await fetch(`${API}/login`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ username, password })
    });

    const data = await res.json();
    if (data.token) {
      localStorage.setItem("token", data.token);
      setToken(data.token);
    } else alert("Invalid credentials");
  };

  // PREDICT
  const predict = async () => {
    const f = {
      age: Number(form.age),
      height: Number(form.height),
      weight: Number(form.weight),
      gender: Number(form.gender),
      ap_hi: Number(form.ap_hi),
      ap_lo: Number(form.ap_lo),
      cholesterol: Number(form.cholesterol),
      gluc: Number(form.gluc),
      smoke: Number(form.smoke),
      alco: Number(form.alco),
      active: Number(form.active)
    };

    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
      },
      body: JSON.stringify(f)
    });

    const data = await res.json();
    setResult(data);
  };

  const getHistory = async () => {
    const res = await fetch(`${API}/history`, {
      headers: { "Authorization": `Bearer ${token}` }
    });
    const data = await res.json();
    setHistory(Array.isArray(data) ? data : []);
  };

  const logout = () => {
    localStorage.removeItem("token");
    setToken(null);
  };

  // 🌟 LOGIN PAGE (ATTRACTIVE)
  if (!token) {
    return (
      <div className="landing">
        <div className="overlay">
          <div className="login-card fade-in">
            <h1>❤️ Heart Health AI</h1>
            <p>Predict cardiovascular risk instantly</p>

            <input placeholder="Username" onChange={(e)=>setUsername(e.target.value)} />
            <input type="password" placeholder="Password" onChange={(e)=>setPassword(e.target.value)} />

            <button onClick={login}>Login</button>
          </div>
        </div>
      </div>
    );
  }

  // DASHBOARD
  return (
    <div className="container fade-in">
      <h1>Heart Disease Prediction Dashboard</h1>

      {/* FORM */}
      <div className="card form-card">
        <h2>Enter Patient Details</h2>

        <div className="form-grid">
          <input name="age" placeholder="Age" onChange={handleChange}/>
          <input name="height" placeholder="Height (cm)" onChange={handleChange}/>
          <input name="weight" placeholder="Weight (kg)" onChange={handleChange}/>
          <input name="ap_hi" placeholder="Systolic BP" onChange={handleChange}/>
          <input name="ap_lo" placeholder="Diastolic BP" onChange={handleChange}/>

          <select name="gender" onChange={handleChange}>
            <option value="1">Male</option>
            <option value="2">Female</option>
          </select>

          <select name="cholesterol" onChange={handleChange}>
            <option value="1">Cholesterol Normal</option>
            <option value="2">Above Normal</option>
            <option value="3">High</option>
          </select>

          <select name="gluc" onChange={handleChange}>
            <option value="1">Glucose Normal</option>
            <option value="2">Above Normal</option>
            <option value="3">High</option>
          </select>

          <select name="smoke" onChange={handleChange}>
            <option value="0">Non-Smoker</option>
            <option value="1">Smoker</option>
          </select>

          <select name="alco" onChange={handleChange}>
            <option value="0">No Alcohol</option>
            <option value="1">Alcohol</option>
          </select>

          <select name="active" onChange={handleChange}>
            <option value="1">Active</option>
            <option value="0">Not Active</option>
          </select>
        </div>

        <button className="primary-btn" onClick={predict}>
          Predict Risk
        </button>
      </div>

      {/* RESULT */}
      {result && (
        <div className="card result">
          <h2>{result.result}</h2>
          <p>Risk: {(result.risk * 100).toFixed(2)}%</p>
        </div>
      )}

      {/* HISTORY */}
      <div className="card">
        <h2>History</h2>
        <button onClick={getHistory}>Load History</button>

        <div className="history-grid">
          {history.map((h,i)=>(
            <div key={i} className="history-card">
              {h.result}
              <span>{(h.risk*100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </div>

      <button className="logout" onClick={logout}>Logout</button>
    </div>
  );
}

export default App;