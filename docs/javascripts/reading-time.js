document$.subscribe(({ body }) => {
  if (!document.querySelector('meta[name="reading-time"]')) return;

  const article = body.querySelector(".md-content__inner");
  if (!article) return;

  const heading = article.querySelector("h1");
  if (!heading) return;

  const words = article.textContent.trim().split(/\s+/).length;
  const minutes = Math.max(1, Math.round(words / 200));

  const badge = document.createElement("div");
  badge.className = "reading-time";
  badge.innerHTML = `<span class="reading-time-icon"></span><span>${minutes} min read</span>`;

  heading.insertAdjacentElement("afterend", badge);
});
