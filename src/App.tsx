import { FiUserPlus, FiUser } from 'react-icons/fi';
import bg from './assets/Camping.svg';
import './App.css';

export default function App() {
  return (
    <div className="hero" style={{ backgroundImage: `url(${bg})` }}>
      <div className="scrim">
        <p className="welcome">Welcome</p>

        <div className="cards">
          <button className="glass cta" type="button">
            <FiUser size={56} aria-hidden />
            <span>Continue as Guest</span>
          </button>
          <button className="glass cta">
            <FiUserPlus size={56} aria-hidden />
            <span>Create an Account</span>
          </button>
        </div>
      </div>
    </div>
  );
}
