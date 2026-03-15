document$.subscribe(({ body }) => {
  const activeLink = body.querySelector(".md-nav__link--active .md-status--draft");
  if (!activeLink) return;

  const article = body.querySelector(".md-content__inner");
  if (!article) return;

  if (article.querySelector(".draft-banner")) return;

  const heading = article.querySelector("h1");
  if (!heading) return;

  const banner = document.createElement("div");
  banner.className = "draft-banner";
  banner.textContent = "This page is a draft and may be incomplete or inaccurate. We're working on finalizing it, but you can speed the process up by ";
  banner.innerHTML += `<a href=https://gitlab.com/risk-metrics/faran/-/blob/main/CONTRIBUTING.md target=_blank>contributing</a>.`;

  heading.insertAdjacentElement("afterend", banner);
});
