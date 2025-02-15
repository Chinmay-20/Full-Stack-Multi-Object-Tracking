// import React, { useState } from "react";
// import axios from "axios";

// const FetchResults: React.FC = () => {
//     const [fileId, setFileId] = useState<string>("");
//     const [stats, setStats] = useState<any>(null);
//     const [error, setError] = useState<string>("");

//     const fetchResults = async () => {
//         if (!fileId) {
//             setError("Please enter a file ID.");
//             return;
//         }

//         setError(""); // Clear previous errors
//         setStats(null); // Reset previous results

//         try {
//             const response = await axios.get(`http://localhost:8000/results/${fileId}`);
//             if (response.data.Error) {
//                 setError(response.data.Error);
//             } else {
//                 setStats(response.data);
//             }
//         } catch (err) {
//             setError("File ID not found or server error.");
//         }
//     };

//     return (
// <div className="container">
//             <h2>Fetch Tracking Statistics</h2>
//             <input
//                 type="text"
//                 placeholder="Enter File ID"
//                 value={fileId}
//                 onChange={(e) => setFileId(e.target.value)}
//             />
//             <button onClick={fetchResults}>Fetch</button>

//             {error && <p className="error">{error}</p>}

//             {stats && (
//                 <div className="stats">
//                     <h3>Tracking Statistics</h3>
//                     <p><strong>File ID:</strong> {stats.file_id}</p>
//                     <p><strong>Total Frames Processed:</strong> {stats.total_frames_processed}</p>
//                     <p><strong>Total Objects Detected:</strong> {stats.total_objects_detected}</p>
//                     <p><strong>Average Objects Per Frame:</strong> {stats.average_objects_per_frame}</p>
//                 </div>
//             )}
//         </div>
//     );
// };

// export default FetchResults;

import React, { useState } from "react";
import axios from "axios";

const FetchResults: React.FC = () => {
    const [fileId, setFileId] = useState<string>("");
    const [stats, setStats] = useState<any>(null);
    const [error, setError] = useState<string>("");

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

    return (
        <div className="container">
            <h2>Fetch Tracking Statistics</h2>
            <input
                type="text"
                placeholder="Enter File ID"
                value={fileId}
                onChange={(e) => setFileId(e.target.value)}
                style={{ width: "100%", padding: "14px", fontSize: "18px" }} 
            />
            <button onClick={fetchResults} style={{ width: "100%", marginTop: "10px" }}>Fetch</button>

            {error && <p className="error">{error}</p>}

            {stats && (
                <div className="stats">
                    <h3>Tracking Statistics</h3>
                    <p><strong>File ID:</strong> {stats.file_id}</p>
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
