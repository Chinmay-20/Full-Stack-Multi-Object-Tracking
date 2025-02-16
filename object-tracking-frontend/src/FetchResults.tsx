import React, { useState } from "react";
import axios from "axios";

const FetchResults: React.FC = () => {
    const [fileId, setFileId] = useState<string>("");
    const [stats, setStats] = useState<any>(null);
    const [error, setError] = useState<string>("");
    const [copied, setCopied] = useState<boolean>(false);

    const fetchResults = async () => {
        if (!fileId) {
            setError("Please enter a file ID.");
            return;
        }

        setError("");
        setStats(null);

        try {
            const response = await axios.get(`http://localhost:8000/results/${fileId}`);
            if (response.data.Error) {
                setError(response.data.Error);
            } else {
                setStats(response.data);
            }
        } catch (err) {
            setError("File ID not found or server error.");
        }
    };

    const handleCopy = () => {
        if (stats?.file_id) {
            navigator.clipboard.writeText(stats.file_id);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    return (
        <div className="container">
            <h2>Fetch Tracking Statistics</h2>
            <div className="input-group">
                <input
                    type="text"
                    placeholder="Enter File ID"
                    value={fileId}
                    onChange={(e) => setFileId(e.target.value)}
                />
                <button onClick={fetchResults}>Fetch</button>
            </div>

            {error && <p className="error">{error}</p>}

            {stats && (
                <div className="stats">
                    <h3>Tracking Statistics</h3>
                    <div className="file-info">
                        <strong>File ID:</strong> 
                        <span className="file-id">{stats.file_id}</span>
                        <button onClick={handleCopy} className="copy-btn">
                            {copied ? "Copied!" : "📋 Copy"}
                        </button>
                    </div>
                    <p><strong>File Name:</strong> {stats.file_name}</p>
                    <p><strong>Total Frames Processed:</strong> {stats.total_frames_processed}</p>
                    <p><strong>Total Objects Detected:</strong> {stats.total_objects_detected}</p>
                    <p><strong>Average Objects Per Frame:</strong> {stats.average_objects_per_frame}</p>
                </div>
            )}
        </div>
    );
};

export default FetchResults;
