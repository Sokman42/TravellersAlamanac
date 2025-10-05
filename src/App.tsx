import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Welcome from "./components/Welcome";
import Menu from "./components/Menu";
import LocationSearch from "./components/LocationSearch";
import Forecast from "./components/Forecast";
import Activities from "./components/Activities";
import Settings from "./components/Settings";
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Welcome />} />
        <Route path="/menu" element={<Menu />} />
        <Route path="/location-search" element={<LocationSearch />} />
        <Route path="/park/:slug" element={<Forecast />} />
        <Route path="/activities" element={<Activities />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
