// import React from "react";
// import { Link } from "react-router-dom";

// const Navbar: React.FC = () => {
//     return (
//         <nav style={styles.navbar}>
//             <ul style={styles.navList}>
//                 <li><Link to="/" style={styles.link}>Home</Link></li>
//                 <li><Link to="/upload" style={styles.link}>Upload File</Link></li>
//                 <li><Link to="/query" style={styles.link}>Query Specific File</Link></li>
//             </ul>
//         </nav>
//     );
// };

// const styles = {
//     navbar: {
//         backgroundColor: "#2c2c2c",
//         padding: "10px",
//         textAlign: "center" as "center",
//     },
//     navList: {
//         listStyle: "none",
//         padding: 0,
//         display: "flex",
//         justifyContent: "center",
//         gap: "20px",
//     },
//     link: {
//         textDecoration: "none",
//         color: "white",
//         fontSize: "18px",
//         fontWeight: "bold",
//     }
// };

// export default Navbar;

import React from "react";
import { Link } from "react-router-dom";

const Navbar: React.FC = () => {
    return (
        <nav>
            <ul>
                <li><Link to="/">Home</Link></li>
                <li><Link to="/upload">Upload File</Link></li>
                <li><Link to="/query">Query Specific File</Link></li>
            </ul>
        </nav>
    );
};

export default Navbar;
