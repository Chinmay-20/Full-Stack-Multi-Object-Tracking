import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Navbar from "./Navbar";
import Home from "./Home";
import UploadComponent from "./UploadComponent";
import FetchResults from "./FetchResults";
import ViewVideo from "./ViewVideo";
import "./index.css";

const App: React.FC = () => {
    return (
        <Router>
            <Navbar />
            <div className="container">
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/upload" element={<UploadComponent />} />
                    <Route path="/query" element={<FetchResults />} />
                    <Route path="/view-video" element={<ViewVideo />} />
                </Routes>
            </div>
        </Router>
    );
};

export default App;
