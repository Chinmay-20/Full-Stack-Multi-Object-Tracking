

import React, { useState } from "react";
import axios from "axios";

const UploadComponent: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [response, setResponse] = useState<{ file_id: string, message: string } | null>(null);
    const [error, setError] = useState<string>("");
    const [copied, setCopied] = useState<boolean>(false);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            setFile(event.target.files[0]);
            setResponse(null);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError("Please select a file.");
            return;
        }

        setError("");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await axios.post("http://localhost:8000/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setResponse(res.data);
        } catch (err) {
            setError("Failed to upload. Please try again.");
        }
    };

    const handleCopy = () => {
        if (response?.file_id) {
            navigator.clipboard.writeText(response.file_id);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    return (
        <div className="container">
            <h2>Upload File for Object Tracking</h2>
            <div className="upload-section">
                <input type="file" onChange={handleFileChange} accept="image/*,video/*" />
                <button onClick={handleUpload}>Upload</button>
            </div>

            {error && <p className="error">{error}</p>}

            {response && (
                <div className="upload-status">
                    <h3>Upload Successful 🎉</h3>
                    <p>{response.message}</p>
                    <div className="file-info">
                        <strong>File ID:</strong> 
                        <span className="file-id">{response.file_id}</span>
                        <button onClick={handleCopy} className="copy-btn">
                            {copied ? "Copied!" : "📋 Copy"}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default UploadComponent;
