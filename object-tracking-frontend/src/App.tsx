import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Navbar from "./Navbar";
import Home from "./Home";
import UploadComponent from "./UploadComponent";
import FetchResults from "./FetchResults";

const App: React.FC = () => {
    return (
        <Router>
            <Navbar />
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/upload" element={<UploadComponent />} />
                <Route path="/query" element={<FetchResults />} />
            </Routes>
        </Router>
    );
};

export default App;
