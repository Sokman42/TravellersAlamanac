import { FiUserPlus, FiLogIn } from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import bg from "../assets/Camping.svg";
import "./Welcome.css"; // or reuse your App.css classes

export default function Welcome() {
    const nav = useNavigate();
    return (
        <div className="hero" style={{ backgroundImage: `url(${bg})` }}>
            <div className="scrim">
                <p className="welcome">Welcome</p>
                <div className="cards">
                    <button className="glass cta" onClick={() => nav("/menu")}>
                        <FiLogIn size={56} aria-hidden />
                        <span>Continue as Guest</span>
                    </button>
                    <button className="glass cta" onClick={() => nav("/menu")}>
                        <FiUserPlus size={56} aria-hidden />
                        <span>Create an Account</span>
                    </button>
                </div>
            </div>
        </div>
    );
}
