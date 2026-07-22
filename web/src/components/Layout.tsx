import type { ReactNode } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { HelpCircle, Shrink } from 'lucide-react';
import ProductTour, { startProductTour } from './ProductTour';

const navClass = ({ isActive }: { isActive: boolean }) =>
  isActive ? 'nav-link nav-link--active' : 'nav-link';

/** App chrome: product header + primary nav, with page content rendered as
 * children. Shared by every product page. Clicking the brand returns to the
 * marketing home. Also mounts the first-visit product tour. */
export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <Link
          to="/"
          className="app-header__brand"
          data-tour="brand"
          aria-label="Kompress home"
        >
          <span className="brand-mark" aria-hidden="true">
            <Shrink size={18} strokeWidth={2.4} />
          </span>
          <span className="brand-name">Kompress</span>
        </Link>

        <nav className="app-nav" aria-label="Primary">
          <NavLink to="/dashboard" className={navClass} data-tour="nav-dashboard">
            Dashboard
          </NavLink>
          <NavLink to="/submit" className={navClass} data-tour="nav-submit">
            Submit
          </NavLink>
          <NavLink
            to="/deployments"
            className={navClass}
            data-tour="nav-deployments"
          >
            Deployments
          </NavLink>
        </nav>

        <div className="app-header__actions">
          <button
            type="button"
            className="icon-btn"
            data-tour="take-tour"
            title="Take a tour"
            aria-label="Take a tour"
            onClick={() => startProductTour()}
          >
            <HelpCircle aria-hidden="true" />
            <span>Take a tour</span>
          </button>
        </div>
      </header>

      <main className="app-main">{children}</main>
      <ProductTour />
    </div>
  );
}
