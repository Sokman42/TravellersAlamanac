import "./Activities.css";
import bg from "../assets/Camping.svg";
import { FiSun, FiCloud, FiWind, FiStar } from "react-icons/fi";
import { TbFlame, TbSnowflake } from "react-icons/tb";

type Opt = { title: string; desc: string; Icon: React.ComponentType<{ size?: number }> };

const options: Opt[] = [
    { title: "Clear Skies", desc: "Stargazing • clear nights", Icon: FiStar },
    { title: "Sunny", desc: "For beaches, hikes, day trips", Icon: FiSun },
    { title: "Cool", desc: "Perfect for minimal layers", Icon: FiWind },
    { title: "Heat", desc: "Quick, easy day trips", Icon: TbFlame },
    { title: "Mixed", desc: "Good for quick, easy day trips", Icon: FiCloud },
    { title: "Chilly", desc: "For those who don’t mind extra layers", Icon: TbSnowflake },
];

export default function Activities() {
    return (
        <div className="act-hero" style={{ backgroundImage: `url(${bg})` }}>
            <div className="act-overlay">
                <header className="act-header">
                    <h2>Choose Your Weather</h2>
                    <p>We’ll find the perfect options based on your selected preference.</p>
                </header>

                <ul className="act-grid">
                    {options.map(({ title, desc, Icon }) => (
                        <li key={title}>
                            <button className="act-card">
                                <Icon size={28} aria-hidden />
                                <h3>{title}</h3>
                                <p>{desc}</p>
                            </button>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}
