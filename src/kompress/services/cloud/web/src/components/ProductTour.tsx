// Guided product tour built on driver.js.
//
// Anchors are `data-tour` attributes placed on the layout and pages; a step is
// only included when its anchor is present in the DOM, so the tour adapts to
// whichever page it is started from. Auto-starts once for new visitors
// (localStorage flag) and can be replayed any time from the header button.
import { useEffect } from 'react';
import { driver } from 'driver.js';
import type { DriveStep } from 'driver.js';
import 'driver.js/dist/driver.css';

const TOUR_DONE_KEY = 'kompress_tour_completed';

export const startProductTour = () => {
  // Small delay so route content is painted before we measure anchors.
  setTimeout(() => {
    const steps: DriveStep[] = [
      {
        element: '[data-tour="brand"]',
        popover: {
          title: 'Welcome to Kompress',
          description:
            'The self-serve console for model compression. Submit a model, watch it get compressed and benchmarked, then approve the winner to production. A quick look around —',
          side: 'bottom',
          align: 'start',
        },
      },
    ];

    const addStep = (
      anchor: string,
      title: string,
      description: string,
      side: 'top' | 'right' | 'bottom' | 'left' = 'bottom',
    ) => {
      const el = document.querySelector(`[data-tour="${anchor}"]`);
      if (!el) return;
      steps.push({
        element: `[data-tour="${anchor}"]`,
        popover: { title, description, side, align: 'start' },
        onHighlightStarted: () => {
          document
            .querySelector(`[data-tour="${anchor}"]`)
            ?.scrollIntoView({ block: 'center', behavior: 'instant' });
        },
      });
    };

    addStep(
      'nav-dashboard',
      'Runs dashboard',
      'The live review queue. Every compression job lands here with its size, latency and quality deltas at a glance.',
    );
    addStep(
      'status-tabs',
      'Filter by status',
      'Slice the queue by lifecycle stage — from pending the quality gate through approved or rejected.',
    );
    addStep(
      'runs-table',
      'Compression runs',
      'Each row is one run. Click through for the full plan: per-variant benchmarks, the promotion gate, and the approve / reject decision.',
      'top',
    );
    addStep(
      'submit-job',
      'Submit a job',
      'Point the pipeline at a model and test set — it compresses every applicable way, benchmarks on your target hardware, and gates the result.',
    );
    addStep(
      'nav-deployments',
      'Deployments',
      'The registered-model catalog: production versions, full history, and one-click rollback.',
    );
    addStep(
      'take-tour',
      'Revisit any time',
      'That is the whole loop — submit, review, approve, roll back. Replay this tour from here whenever you need it.',
      'bottom',
    );

    const driverObj = driver({
      showProgress: true,
      allowClose: true,
      popoverClass: 'kp-tour-popover',
      nextBtnText: 'Next',
      prevBtnText: 'Back',
      doneBtnText: 'Done',
      onDestroyed: () => {
        localStorage.setItem(TOUR_DONE_KEY, 'true');
      },
      steps,
    });

    driverObj.drive();
  }, 300);
};

/** Mount once in the layout: auto-runs the tour for first-time visitors.
 * `?tour=off` suppresses the auto-start (handy for demos and screenshots). */
export default function ProductTour() {
  useEffect(() => {
    const timer = setTimeout(() => {
      if (localStorage.getItem(TOUR_DONE_KEY) === 'true') return;
      if (new URLSearchParams(window.location.search).get('tour') === 'off') return;
      startProductTour();
    }, 600);
    return () => clearTimeout(timer);
  }, []);

  return null;
}
