import UploadForm from "@/components/UploadForm";
import ForgeryResult from "@/components/ForgeryResult";
import MetricsPanel from "@/components/MetricsPanel";

export default function Home() {
  return (
    <main className="p-6 max-w-4xl mx-auto font-sans">
      <h1 className="text-3xl font-bold mb-6">Copy-Move Forgery Detection</h1>
      <UploadForm />
      <ForgeryResult />
      <MetricsPanel />
    </main>
  );
}
