import bg from "../assets/Camping.svg";
import "./Settings.css";

export default function Settings() {
    return (
        <div className="hero" style={{ backgroundImage: `url(${bg})` }}>
            <div className="scrim" style={{ display: "grid", placeItems: "center" }}>
                <p style={{ color: "#fff", opacity: .9 }}>
                    It’s a settings page. We’ll figure out something on the job.
                </p>
            </div>
        </div>
    );
}
