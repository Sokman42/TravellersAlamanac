import "./Forecast.css";
import { useLocation, useParams, Link } from "react-router-dom";
import { FiSun, FiCloud, FiUmbrella, FiWind } from "react-icons/fi";
import fallbackBg from "../assets/Camping.svg";

type Park = { name: string; image: string };

export default function Forecast() {
    const { state } = useLocation() as { state?: { park?: Park } };
    const { slug } = useParams();
    const park = state?.park; // simple: we rely on passed state

    const weeks = [
        { range: "April 20 – 26", hi: 28, lo: 14, note: "Hot Week Ahead", Icon: FiUmbrella },
        { range: "May 18 – 24", hi: 19, lo: 12, note: "Clear Nights", Icon: FiWind },
        { range: "June 8 – 14", hi: 26, lo: 16, note: "Sunny Days", Icon: FiSun },
        { range: "July 20 – 25", hi: 29, lo: 18, note: "Warm + Dry", Icon: FiCloud },
    ];

    return (
        <div
            className="forecast-hero"
            style={{ backgroundImage: `url(${park?.image ?? fallbackBg})` }}
        >
            <div className="overlay">
                <header className="top">
                    <h2>Forecast: <span className="accent">{park?.name ?? slug}</span></h2>
                    <Link to="/location-search" className="back">← Back</Link>
                </header>
                <p className="subtitle">Upcoming highlight weeks based on your preferences</p>

                <div className="week-row" id="row">
                    {weeks.map(({ range, hi, lo, note, Icon }) => (
                        <div key={range} className="week-card glass">
                            <div className="range">{range}</div>
                            <Icon size={36} aria-hidden />
                            <div className="temps">High {hi}° | Low {lo}°</div>
                            <div className="note">{note}</div>
                            <button className="book">Book on Website →</button>
                        </div>
                    ))}
                </div>

                <div className="arrows">
                    <button onClick={() => document.getElementById("row")!.scrollBy({ left: -320, behavior: "smooth" })}>‹</button>
                    <button onClick={() => document.getElementById("row")!.scrollBy({ left: 320, behavior: "smooth" })}>›</button>
                </div>
            </div>
        </div>
    );
}
