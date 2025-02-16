import React, { useEffect, useState } from "react";
import axios from "axios";

const Home: React.FC = () => {
    const [files, setFiles] = useState<{ file_id: string, file_name: string }[]>([]);
    const [error, setError] = useState<string>("");
    const [copiedId, setCopiedId] = useState<string | null>(null);

    useEffect(() => {
        fetchFiles();
    }, []);

    const fetchFiles = () => {
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
    };

    const handleDelete = async (fileId: string) => {
        try {
            await axios.delete(`http://localhost:8000/delete/${fileId}`);
            setFiles(files.filter(file => file.file_id !== fileId)); // Remove from UI
        } catch (err) {
            alert("Failed to delete file.");
        }
    };

    const handleCopy = (fileId: string) => {
        navigator.clipboard.writeText(fileId);
        setCopiedId(fileId);
        setTimeout(() => setCopiedId(null), 2000); // Reset "Copied!" after 2 sec
    };

    return (
        <div className="container">
            <h2>Uploaded Files</h2>
            {error && <p className="error">{error}</p>}
            <div style={{ display: "grid", gap: "10px", textAlign: "left" }}>
                {files.map((file) => (
                    <div key={file.file_id} className="file-card">
                        <strong>{file.file_name}</strong>
                        <div className="file-actions">
                            <p 
                                className="file-id" 
                                title="Click to copy"
                                onClick={() => handleCopy(file.file_id)}
                                style={{ cursor: "pointer" }}
                            >
                                ID: {file.file_id} 📋
                            </p>
                            {copiedId === file.file_id && (
                                <span style={{ fontSize: "12px", color: "#38bdf8" }}>Copied!</span>
                            )}
                            <button 
                                className="delete-btn"
                                onClick={() => handleDelete(file.file_id)}
                            >
                                Delete
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Home;

