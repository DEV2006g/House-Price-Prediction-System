/**
 * Interactive Client Logic for House Price Prediction System
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const priceForm = document.getElementById("price-form");
  const presetCards = document.querySelectorAll(".preset-card");

  // Output displays
  const displayPrice = document.getElementById("display-price");
  const displayRange = document.getElementById("range-values");
  const displaySqft = document.getElementById("display-sqft");
  const driverQual = document.getElementById("driver-qual");
  const driverSqft = document.getElementById("driver-sqft");
  const driverAge = document.getElementById("driver-age");
  const driverLoc = document.getElementById("driver-loc");

  // 1. Tab Switching
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.dataset.tab;
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const activePane = document.getElementById(`tab-${targetTab}`);
      if (activePane) activePane.classList.add("active");
    });
  });

  // 2. Form submission / calculation
  async function calculateValuation() {
    const formData = new FormData(priceForm);
    const payload = {};
    formData.forEach((val, key) => {
      payload[key] = val;
    });

    // Update Drivers sidebar info
    if (driverQual) driverQual.textContent = `${payload.OverallQual || 7}/10`;
    if (driverSqft) {
      const sqft = parseInt(payload.GrLivArea || 0, 10);
      driverSqft.textContent = `${sqft.toLocaleString()} sq ft`;
    }
    if (driverAge) driverAge.textContent = `Built ${payload.YearBuilt || 2004}`;
    if (driverLoc) driverLoc.textContent = payload.Neighborhood || "CollgCr";

    try {
      if (displayPrice) displayPrice.style.opacity = "0.5";
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (data.success && data.result) {
        const r = data.result;
        animateNumber(displayPrice, r.predicted_price);
        if (displayRange) displayRange.textContent = r.formatted_range;
        if (displaySqft) displaySqft.textContent = r.formatted_sqft_price;
      } else {
        alert("Error predicting valuation: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      console.error("Valuation request failed", err);
    } finally {
      if (displayPrice) displayPrice.style.opacity = "1";
    }
  }

  // Smooth number counter animation
  function animateNumber(element, targetVal) {
    if (!element) return;
    const duration = 400;
    const startVal = parseInt(element.textContent.replace(/[^0-9]/g, ""), 10) || targetVal;
    const startTime = performance.now();

    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      const current = Math.floor(startVal + (targetVal - startVal) * easeProgress);
      element.textContent = `$${current.toLocaleString()}`;

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        element.textContent = `$${Math.round(targetVal).toLocaleString()}`;
      }
    }
    requestAnimationFrame(step);
  }

  // 3. Form event listeners
  if (priceForm) {
    priceForm.addEventListener("submit", (e) => {
      e.preventDefault();
      calculateValuation();
    });

    // Auto-recalculate on change of major features
    const autoRecalcInputs = ["OverallQual", "OverallCond", "GrLivArea", "TotalBsmtSF", "YearBuilt", "Neighborhood"];
    autoRecalcInputs.forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("change", () => {
          calculateValuation();
        });
      }
    });
  }

  // 4. Quick Preset Selectors
  presetCards.forEach((card) => {
    card.addEventListener("click", async () => {
      const presetKey = card.dataset.preset;
      try {
        const res = await fetch("/api/presets");
        const presets = await res.json();
        const preset = presets[presetKey];
        if (!preset) return;

        // Populate fields
        for (const [key, val] of Object.entries(preset.data)) {
          const input = document.getElementById(key);
          if (input) {
            input.value = val;
            // Update output badges for range sliders
            if (input.type === "range" && input.nextElementSibling) {
              input.nextElementSibling.value = `${val} / 10`;
            }
          }
        }

        // Highlight selected preset
        presetCards.forEach((c) => (c.style.borderColor = "var(--border-color)"));
        card.style.borderColor = "var(--primary)";

        // Run valuation
        calculateValuation();
      } catch (err) {
        console.error("Failed to load preset", err);
      }
    });
  });

  // Calculate initial valuation on load
  setTimeout(calculateValuation, 200);
});
