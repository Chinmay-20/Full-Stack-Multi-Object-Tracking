import React from "react";
import { Link } from "react-router-dom";

const Navbar: React.FC = () => {
    return (
        <nav>
            <ul>
                <li><Link to="/">Home</Link></li>
                <li><Link to="/upload">Upload File</Link></li>
                <li><Link to="/query">Query Specific File</Link></li>
                <li><Link to="/view-video">View Processed Video</Link></li>
            </ul>
        </nav>
    );
};

export default Navbar;
