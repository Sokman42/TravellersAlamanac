import './LocationSearch.css';
import bg from '../assets/Camping.svg';
import { Link } from 'react-router-dom';
import { FiCloud } from 'react-icons/fi';
import { MdDirectionsBike } from 'react-icons/md';
import { FaCampground } from 'react-icons/fa';
import BrontePark from '../assets/BrontePark.svg';
import RockPark from '../assets/RockPark.svg';
import ForksPark from '../assets/ForksPark.svg';

// slug helper (outside the component is fine)
const slug = (s: string) =>
    s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');

export default function LocationSearch() {
    const parks = [
        { name: 'Bronte Creek Provincial Park', distance: '12 km', image: BrontePark, weather: 'Mixed Weather', info: 'Great for Stargazing' },
        { name: 'Rock Point Provincial Park', distance: '23 km', image: RockPark, weather: 'Sunny Days Ahead', info: 'Enjoy the beach vibes' },
        { name: 'Forks of the Credit', distance: '44 km', image: ForksPark, weather: 'Clear and Cool', info: 'Great for Stargazing' },
    ];

    return (
        <div className="location-page" style={{ backgroundImage: `url(${bg})` }}>
            <div className="overlay">
                <div className="header">
                    <h2>Parks Near You</h2>
                    <p>Let's find the perfect campground for you</p>
                    <small>[Distance and weather-type filters and stuff here]</small>
                </div>

                <div className="card-grid">
                    {parks.map((p) => (
                        <Link
                            key={p.name}
                            to={`/park/${slug(p.name)}`}        // route with param
                            state={{ park: p }}                  // pass the park object
                            className="park-card glass"          // keep your existing styles
                        >
                            <img src={p.image} alt={p.name} className="park-image" />
                            <div className="park-info">
                                <h3>{p.name}</h3>
                                <p className="distance">{p.distance}</p>
                                <div className="icons"><FaCampground /><FiCloud /><MdDirectionsBike /></div>
                                <button className="view-btn">View Available Weeks <span>→</span></button>
                                <p className="extra">{p.info}</p>
                            </div>
                        </Link>
                    ))}
                </div>
            </div>
        </div>
    );
}
