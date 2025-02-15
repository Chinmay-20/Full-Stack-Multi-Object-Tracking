// import React, { useEffect, useState } from "react";
// import axios from "axios";

// const Home: React.FC = () => {
//     const [files, setFiles] = useState<{ file_id: string, file_name: string }[]>([]);
//     const [error, setError] = useState<string>("");

//     useEffect(() => {
//         axios.get("http://localhost:8000/files")
//             .then((response) => {
//                 if (response.data.message) {
//                     setError(response.data.message);
//                 } else {
//                     setFiles(response.data.files);
//                 }
//             })
//             .catch(() => {
//                 setError("Failed to fetch files.");
//             });
//     }, []);

//     return (
//         <div className="container">
//             <h2>Uploaded Files</h2>
//             {error && <p className="error">{error}</p>}
//             <ul>
//                 {files.map((file) => (
//                     <li key={file.file_id}>
//                         <strong>{file.file_name}</strong> - {file.file_id}
//                     </li>
//                 ))}
//             </ul>
//         </div>
//     );
// };

// export default Home;

import React, { useEffect, useState } from "react";
import axios from "axios";

const Home: React.FC = () => {
    const [files, setFiles] = useState<{ file_id: string, file_name: string }[]>([]);
    const [error, setError] = useState<string>("");
    const [copiedId, setCopiedId] = useState<string | null>(null);

    useEffect(() => {
        axios.get("http://localhost:8000/files")
            .then((response) => {
                if (response.data.message) {
                    setError(response.data.message);
                } else {
                    setFiles(response.data.files);
                }
            })
            .catch(() => {
                setError("Failed to fetch files.");
            });
    }, []);

    // Copy function
    const handleCopy = (fileId: string) => {
        navigator.clipboard.writeText(fileId);
        setCopiedId(fileId);

        setTimeout(() => setCopiedId(null), 2000); // Reset copy status after 2 sec
    };

    return (
        <div className="container">
            <h2>Uploaded Files</h2>
            {error && <p className="error">{error}</p>}
            <div style={{ display: "grid", gap: "10px", textAlign: "left" }}>
                {files.map((file) => (
                    <div key={file.file_id} style={{
                        background: "#2d2d33",
                        padding: "12px",
                        borderRadius: "6px",
                    }}>
                        <strong>{file.file_name}</strong>
                        <p 
                            style={{ fontSize: "14px", color: "#a1a1aa", cursor: "pointer" }} 
                            onClick={() => handleCopy(file.file_id)}
                            title="Click to copy"
                        >
                            ID: {file.file_id} 📋
                        </p>
                        {copiedId === file.file_id && (
                            <span style={{ fontSize: "12px", color: "#38bdf8" }}>Copied!</span>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Home;
