// ==========================================================================
// SwadeshiKart - Smart Search Frontend Script
// ==========================================================================

// ---------------- Splash Screen Handler ----------------
(function () {
  const splash = document.getElementById("splash-screen");
  if (!splash) return;

  // Splash slides UP at 1.6s (animation-delay) over 0.55s.
  // Listen for animationend to remove it from the DOM completely.
  splash.addEventListener("animationend", () => {
    splash.classList.add("hidden");
  });
})();

const searchInput = document.getElementById("search");
const clearBtn = document.getElementById("clear-search");
const suggestionsList = document.getElementById("suggestions");
const messageBox = document.getElementById("message");
const categoryBanner = document.getElementById("category-banner");
const bannerIcon = document.getElementById("banner-icon");
const bannerTitle = document.getElementById("banner-title");
const bannerTagline = document.getElementById("banner-tagline");
const savingsText = document.getElementById("savings-text");
const resultsSection = document.getElementById("results");
const foreignBox = document.getElementById("foreign-list");
const indianBox = document.getElementById("indian-list");
const foreignCount = document.getElementById("foreign-count");
const indianCount = document.getElementById("indian-count");
const categoryPills = document.querySelectorAll(".cat-pill");

const mobileTabsBar = document.getElementById("mobile-view-tabs");
const mobileTabBtns = document.querySelectorAll(".mobile-tab-btn");
const mobileIndianCount = document.getElementById("m-tab-indian-count");
const mobileForeignCount = document.getElementById("m-tab-foreign-count");

let shownSuggestions = [];
let activeIndex = -1;
let debounceTimer = null;

// ---------------- Dynamic Placeholder Rotation ----------------
const PLACEHOLDERS = [
  'Search "water", "mineral water"...',
  'Search "chips", "namkeen", "wafers"...',
  'Search "biscuits", "cookies", "crackers"...',
  'Search "detergent", "floor cleaner", "cleaning"...',
  'Search "shampoo", "hair care", "herbal"...',
  'Search "dishwash gel", "Vim", "Exo"...',
  'Search "Parle", "Tata", "Ghadi", "Himalaya"...',
  'Search "Oreo", "Surf Excel", "Lizol", "L\'Oreal"...',
];
let placeholderIndex = 0;

setInterval(() => {
  if (document.activeElement !== searchInput && !searchInput.value) {
    placeholderIndex = (placeholderIndex + 1) % PLACEHOLDERS.length;
    searchInput.placeholder = PLACEHOLDERS[placeholderIndex];
  }
}, 3000);

// ---------------- Suggestions Dropdown (Smart & Typo-Tolerant) ----------------
function fetchSuggestions(query) {
  clearTimeout(debounceTimer);
  clearBtn.hidden = query.length === 0;

  debounceTimer = setTimeout(() => {
    fetch("/api/suggestions?q=" + encodeURIComponent(query) + "&limit=8")
      .then(res => {
        if (!res.ok) throw new Error("HTTP error " + res.status);
        return res.json();
      })
      .then(items => {
        renderSuggestionsUI(items, query);
      })
      .catch(() => {
        // Fallback to local filtering if network drops
        fallbackLocalSuggestions(query);
      });
  }, 120);
}

function fallbackLocalSuggestions(query) {
  const q = query.trim().toLowerCase();
  let filtered = [];
  if (!q) {
    filtered = typeof SUGGESTIONS !== "undefined" ? SUGGESTIONS.filter(s => s.isPrimary) : [];
  } else if (typeof SUGGESTIONS !== "undefined") {
    filtered = SUGGESTIONS.filter(s => {
      if (s.label.toLowerCase().includes(q)) return true;
      if (s.categoryName && s.categoryName.toLowerCase().includes(q)) return true;
      if (Array.isArray(s.keywords) && s.keywords.some(k => k.toLowerCase().includes(q))) return true;
      return false;
    });
  }
  renderSuggestionsUI(filtered, query);
}

function renderSuggestionsUI(items, query) {
  shownSuggestions = items || [];
  activeIndex = -1;
  suggestionsList.innerHTML = "";

  if (shownSuggestions.length === 0) {
    suggestionsList.hidden = true;
    searchInput.setAttribute("aria-expanded", "false");
    return;
  }

  shownSuggestions.forEach((s, idx) => {
    const li = document.createElement("li");
    li.className = "suggestion-item";
    li.setAttribute("role", "option");
    li.id = `suggestion-${idx}`;

    const left = document.createElement("div");
    left.className = "suggestion-left";

    const icon = document.createElement("span");
    icon.className = "suggestion-icon";
    icon.textContent = s.icon || "🛒";

    const title = document.createElement("span");
    title.className = "suggestion-title";
    title.textContent = s.label;

    left.append(icon, title);

    const tag = document.createElement("span");
    tag.className = "suggestion-type-tag";
    tag.textContent = s.type || (s.isPrimary ? "Category" : (s.categoryName || "Item"));

    if (s.type && s.type.includes("Foreign")) {
      tag.style.background = "#fff1f2";
      tag.style.color = "#dc2626";
    } else if (s.type === "Correction") {
      tag.style.background = "#fef9c3";
      tag.style.color = "#854d0e";
    }

    li.append(left, tag);

    li.addEventListener("mousedown", e => {
      e.preventDefault();
      chooseSuggestion(s);
    });

    suggestionsList.appendChild(li);
  });

  suggestionsList.hidden = false;
  searchInput.setAttribute("aria-expanded", "true");
}

function chooseSuggestion(s) {
  const searchTerm = s.query || s.label;
  searchInput.value = searchTerm;
  clearBtn.hidden = false;
  suggestionsList.hidden = true;
  searchInput.setAttribute("aria-expanded", "false");
  hideMessage();

  if (s.type === "Category" && s.category) {
    loadSmartSearch("", s.category);
  } else {
    loadSmartSearch(searchTerm);
  }
}

// ---------------- Load Comparison Results (Smart Search) ----------------
function loadSmartSearch(query = "", categoryId = "") {
  // Update active category pill
  categoryPills.forEach(pill => {
    pill.classList.toggle("active", Boolean(categoryId && pill.dataset.catId === categoryId));
  });

  let url = "/search?";
  if (query) {
    url += "q=" + encodeURIComponent(query);
    if (categoryId) url += "&category=" + encodeURIComponent(categoryId);
  } else if (categoryId) {
    url += "category=" + encodeURIComponent(categoryId);
  } else {
    url += "category=water";
  }

  fetch(url)
    .then(res => {
      if (!res.ok) throw new Error("HTTP error " + res.status);
      return res.json();
    })
    .then(data => {
      if (data.total_matches === 0) {
        categoryBanner.hidden = true;
        resultsSection.hidden = true;
        if (mobileTabsBar) mobileTabsBar.hidden = true;
        showMessage(`No matching products found for "${query || categoryId}". Try searching for water, chips, biscuits, cleaning, shampoo, or brands like Surf Excel, Ghadi, Parle, Tata.`);
        return;
      }

      hideMessage();
      renderCategoryBanner(data.category, data.stats, data.message);
      renderCardsList(foreignBox, data.foreign, "foreign", data.stats);
      renderCardsList(indianBox, data.indian, "indian", data.stats);

      foreignCount.textContent = `${data.foreign.length} Products`;
      indianCount.textContent = `${data.indian.length} Alternatives`;

      if (mobileIndianCount) mobileIndianCount.textContent = data.indian.length;
      if (mobileForeignCount) mobileForeignCount.textContent = data.foreign.length;
      if (mobileTabsBar) mobileTabsBar.hidden = false;

      categoryBanner.hidden = false;
      resultsSection.hidden = false;

      // Re-trigger column pop-up animation on every render
      const colForeign = resultsSection.querySelector(".col-foreign");
      const colIndian  = resultsSection.querySelector(".col-indian");
      [colForeign, colIndian].forEach(col => {
        if (!col) return;
        col.classList.remove("col-pop", "col-pop-left", "col-pop-right");
        // Force reflow so the browser resets the animation
        void col.offsetWidth;
      });
      if (colForeign) colForeign.classList.add("col-pop", "col-pop-left");
      if (colIndian)  colIndian.classList.add("col-pop", "col-pop-right");
    })
    .catch(err => {
      console.error("Fetch error:", err);
      showMessage("Could not load products. Please check connection and try again.");
    });
}

function renderCategoryBanner(categoryMeta, stats, noticeMessage) {
  bannerIcon.textContent = categoryMeta.icon || "🛒";
  bannerTitle.textContent = categoryMeta.name || "Category";
  
  if (noticeMessage) {
    bannerTagline.textContent = noticeMessage;
  } else {
    bannerTagline.textContent = categoryMeta.tagline || "Compare foreign brands with Indian domestic alternatives";
  }

  if (stats && stats.savings_pct > 0) {
    const savingsDiff = Math.max(Math.round(stats.avg_foreign - stats.avg_indian), 0);
    savingsText.innerHTML = `Switch to Indian brands & save an average of <strong>${stats.savings_pct}%</strong> (₹${savingsDiff} avg savings)!`;
  } else {
    savingsText.innerHTML = `Compare foreign brands with 100% Indian alternatives`;
  }
}

// ---------------- Render Product Cards ----------------
function renderCardsList(container, items, type, stats) {
  container.innerHTML = "";

  if (!items || items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-col";
    empty.textContent = type === "foreign"
      ? "No foreign brands found."
      : "No Indian alternatives found.";
    container.appendChild(empty);
    return;
  }

  items.forEach((item, idx) => {
    const card = createProductCard(item, type, stats);
    // Stagger each card: 55ms apart, pop-up spring animation driven by JS
    card.style.animationDelay = `${idx * 0.055}s`;
    container.appendChild(card);
  });
}

function createProductCard(p, type, stats) {
  const card = document.createElement("article");
  card.className = type === "foreign" ? "product-card card-foreign" : "product-card card-indian";

  // Image wrapper
  const imgWrap = document.createElement("div");
  imgWrap.className = "card-img-wrap";

  const img = document.createElement("img");
  img.className = "card-img";
  img.alt = p.name;
  img.loading = "lazy";

  const showFallback = () => {
    img.onerror = null;
    img.style.display = "none";
    const fallback = document.createElement("div");
    fallback.style.cssText = "display:grid;place-items:center;width:100%;height:100%;font-size:1.6rem;font-weight:800;color:#9ca3af;background:#f1f5f9;";
    fallback.textContent = p.brand ? p.brand.charAt(0) : "🛒";
    imgWrap.appendChild(fallback);
  };

  // Image search order: PNG -> JPG -> SVG -> Fallback
  const isExternalUrl = p.image && (p.image.startsWith("http://") || p.image.startsWith("https://"));

  if (isExternalUrl) {
    img.src = p.image;
    img.onerror = showFallback;
  } else {
    const baseName = p.image ? p.image.replace(/\.[^/.]+$/, "") : (p.id ? String(p.id) : "placeholder");
    const formats = ["png", "jpg", "svg"];
    let formatIndex = 0;

    img.src = `/static/images/${baseName}.${formats[formatIndex]}`;

    img.onerror = () => {
      formatIndex++;
      if (formatIndex < formats.length) {
        img.src = `/static/images/${baseName}.${formats[formatIndex]}`;
      } else {
        showFallback();
      }
    };
  }

  const originTag = document.createElement("span");
  if (type === "foreign") {
    originTag.className = "card-origin-tag origin-tag-foreign";
    originTag.textContent = `⚠️ ${p.country}`;
  } else {
    originTag.className = "card-origin-tag origin-tag-indian";
    originTag.textContent = `🇮🇳 ${p.country}`;
  }

  imgWrap.append(img, originTag);

  // Info Section
  const info = document.createElement("div");
  info.className = "card-info";

  const top = document.createElement("div");
  top.className = "card-top";

  const brand = document.createElement("div");
  brand.className = "card-brand";
  brand.textContent = p.brand;

  const title = document.createElement("h4");
  title.className = "card-title";
  title.textContent = p.name;

  const qty = document.createElement("div");
  qty.className = "card-qty";
  qty.textContent = `Pack: ${p.quantity}`;

  top.append(brand, title, qty);

  // Bottom Section: Price & Action
  const bottom = document.createElement("div");
  bottom.className = "card-bottom";

  const priceWrap = document.createElement("div");
  priceWrap.className = "price-wrap";

  const price = document.createElement("span");
  price.className = "card-price";
  price.textContent = `₹${p.price}`;
  priceWrap.appendChild(price);

  if (type === "foreign") {
    const warnChip = document.createElement("span");
    warnChip.className = "warning-chip";
    warnChip.textContent = "⚠️ High Markup";
    priceWrap.appendChild(warnChip);
  } else if (type === "indian" && stats && stats.avg_foreign > 0) {
    const itemPrice = parseFloat(p.price) || 0;
    if (itemPrice < stats.avg_foreign) {
      const diff = Math.round(stats.avg_foreign - itemPrice);
      const chip = document.createElement("span");
      chip.className = "savings-chip";
      chip.textContent = `Save ~₹${diff}`;
      priceWrap.appendChild(chip);
    }
  }

  // Direct Product Link Button
  const btn = document.createElement("a");
  btn.href = p.link || "#";
  btn.target = "_blank";
  btn.rel = "noopener noreferrer";

  if (type === "foreign") {
    btn.className = "btn-buy btn-buy-foreign";
    btn.textContent = "View Item ↗";
    btn.title = `Check ${p.name} on Blinkit`;
  } else {
    btn.className = "btn-buy btn-buy-indian";
    btn.textContent = "Buy on Blinkit ↗";
    btn.title = `Buy ${p.name} on Blinkit`;
  }

  bottom.append(priceWrap, btn);
  info.append(top, bottom);
  card.append(imgWrap, info);

  return card;
}

// ---------------- Alerts / Helper ----------------
function showMessage(text) {
  messageBox.textContent = text;
  messageBox.hidden = false;
}

function hideMessage() {
  messageBox.hidden = true;
}

// ---------------- Keyboard & Search Events ----------------
function setActiveSuggestion(index) {
  const items = suggestionsList.querySelectorAll(".suggestion-item");
  items.forEach((item, idx) => {
    item.classList.toggle("active", idx === index);
    if (idx === index) {
      item.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });
  activeIndex = index;
}

searchInput.addEventListener("input", () => {
  fetchSuggestions(searchInput.value);
});

searchInput.addEventListener("focus", () => {
  fetchSuggestions(searchInput.value);
});

searchInput.addEventListener("blur", () => {
  setTimeout(() => {
    suggestionsList.hidden = true;
    searchInput.setAttribute("aria-expanded", "false");
  }, 220);
});

clearBtn.addEventListener("click", () => {
  searchInput.value = "";
  clearBtn.hidden = true;
  suggestionsList.hidden = true;
  hideMessage();
  searchInput.focus();
  // Reset to default first category
  if (typeof CATEGORIES !== "undefined" && CATEGORIES.length > 0) {
    loadSmartSearch("", CATEGORIES[0].id);
  }
});

searchInput.addEventListener("keydown", e => {
  if (e.key === "ArrowDown" && shownSuggestions.length > 0) {
    e.preventDefault();
    const next = activeIndex < shownSuggestions.length - 1 ? activeIndex + 1 : 0;
    setActiveSuggestion(next);
  } else if (e.key === "ArrowUp" && shownSuggestions.length > 0) {
    e.preventDefault();
    const prev = activeIndex > 0 ? activeIndex - 1 : shownSuggestions.length - 1;
    setActiveSuggestion(prev);
  } else if (e.key === "Enter") {
    e.preventDefault();
    suggestionsList.hidden = true;
    searchInput.setAttribute("aria-expanded", "false");

    if (activeIndex >= 0 && shownSuggestions[activeIndex]) {
      chooseSuggestion(shownSuggestions[activeIndex]);
    } else {
      const q = searchInput.value.trim();
      if (q) {
        clearBtn.hidden = false;
        loadSmartSearch(q);
      }
    }
  } else if (e.key === "Escape") {
    suggestionsList.hidden = true;
    searchInput.setAttribute("aria-expanded", "false");
  }
});

// ---------------- Category Pills Event Listeners ----------------
categoryPills.forEach(pill => {
  pill.addEventListener("click", () => {
    const catId = pill.dataset.catId;
    const catMeta = typeof CATEGORIES !== "undefined" ? CATEGORIES.find(c => c.id === catId) : null;
    if (catMeta) {
      searchInput.value = catMeta.name;
      clearBtn.hidden = false;
    }
    loadSmartSearch("", catId);
  });
});

// ---------------- Mobile View Switcher Event Listeners ----------------
mobileTabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    const view = btn.dataset.view;
    mobileTabBtns.forEach(b => b.classList.toggle("active", b === btn));
    resultsSection.classList.remove("show-indian-only", "show-foreign-only");
    if (view === "indian") {
      resultsSection.classList.add("show-indian-only");
    } else if (view === "foreign") {
      resultsSection.classList.add("show-foreign-only");
    }
  });
});

// Auto-load first category immediately on page open
document.addEventListener("DOMContentLoaded", () => {
  if (typeof CATEGORIES !== "undefined" && CATEGORIES.length > 0) {
    loadSmartSearch("", CATEGORIES[0].id);
  }
});
