import { useEffect, useState } from "react";
import axios from "axios";

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    axios.get("http://localhost:5000/metrics").then((res) => {
      setMetrics(res.data);
    });
  }, []);

  return (
    <div className="my-8">
      <h2 className="text-xl font-bold mb-4">Model Metrics</h2>
      {metrics ? (
        <div className="bg-gray-100 p-4 rounded shadow text-sm">
          <p><strong>Accuracy:</strong> {metrics.accuracy.toFixed(4)}</p>
          <p><strong>Precision:</strong> {metrics.precision.toFixed(4)}</p>
          <p><strong>Recall:</strong> {metrics.recall.toFixed(4)}</p>
          <p><strong>F1 Score:</strong> {metrics.f1_score.toFixed(4)}</p>
          <p><strong>Confusion Matrix:</strong> [{metrics.confusion_matrix[0].join(", ")}], [{metrics.confusion_matrix[1].join(", ")}]</p>
          <img
            src="http://localhost:5000/stats"
            alt="Training Stats"
            className="mt-4 border"
          />
        </div>
      ) : (
        <p className="text-gray-500">Loading metrics...</p>
      )}
    </div>
  );
}
