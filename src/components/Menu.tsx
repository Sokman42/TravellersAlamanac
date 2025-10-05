import { FiFeather, FiSettings } from "react-icons/fi";
import { TbFlame } from "react-icons/tb";
import { Link } from "react-router-dom";
import bg from "../assets/Camping.svg";
import "./Menu.css";

export default function Menu() {
    return (
        <div className="hero" style={{ backgroundImage: `url(${bg})` }}>
            <div className="scrim">
                <header className="topbar">
                    <h1>Hello, <span className="accent">Camper.</span></h1>
                    <div className="temp">Current temp <strong>20°c</strong></div>
                </header>

                <ul className="menuGrid">
                    <li>
                        <Link to="/activities" className="menuCard">
                            <FiFeather size={44} />
                            <h3>Activities</h3>
                            <div className="divider" />
                            <p>Choose what you would like to do</p>
                        </Link>
                    </li>

                    <li>
                        <Link to="/location-search" className="menuCard">
                            <TbFlame size={44} aria-hidden />
                            <h3>Adventure</h3>
                            <div className="divider" />
                            <p>Find parks and campgrounds near you</p>
                        </Link>
                    </li>
                    <li>
                        <Link to="/settings" className="menuCard">
                            <FiSettings size={44} aria-hidden />
                            <h3>Settings</h3>
                            <div className="divider" />
                            <p>Set preferences & notifications</p>
                        </Link>
                    </li>                </ul>

                <div style={{ marginTop: 16, textAlign: "center" }}>
                    <Link to="/" style={{ color: "#fff" }}>← Back to Welcome</Link>
                </div>
            </div>
        </div>
    );
}
