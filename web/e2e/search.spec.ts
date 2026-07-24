import { expect, test } from "@playwright/test";

test("submits a scene search and renders the matching passage", async ({
  page
}) => {
  await page.route("**/api/search", async (route) => {
    expect(route.request().postDataJSON()).toMatchObject({
      mode: "lexical",
      page: 1,
      page_size: 10
    });
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        query: "blue flower at the Wall",
        result_count: 1,
        duration_ms: 82,
        cached: false,
        results: [
          {
            rank: 1,
            score: 0.81,
            chunk: {
              id: "acok-048-daenerys-4-c001",
              chapter_id: "acok-048-daenerys-4",
              chapter_title: "DAENERYS IV",
              chapter_sequence: 48,
              pov: "DAENERYS",
              pov_ordinal: 4,
              chunk_ordinal: 1,
              word_start: 0,
              word_end: 180,
              book_id: "acok",
              book_title: "A Clash of Kings",
              book_sequence: 2
            },
            context_before: "She remembered the dream.",
            excerpt: "A blue flower grew from a chink in a wall of ice.",
            context_after: "Ice rose around them.",
            retrieval: { mode: "hierarchical-hybrid" }
          }
        ]
      })
    });
  });

  await page.goto("/");
  await page
    .getByPlaceholder("Tyrion demands a trial by combat at the Eyrie")
    .fill("blue flower at the Wall");
  await expect(page.getByRole("radio", { name: "Semantic" })).toBeChecked();
  await page.getByText("Need exact words? Try lexical search").click();
  await expect(page.getByRole("radio", { name: "Lexical" })).toBeChecked();
  await page.getByRole("button", { name: "Search" }).click();

  await expect(
    page.getByText("A blue flower grew from a chink in a wall of ice.")
  ).toBeVisible();
  await expect(page).toHaveURL(/\?q=blue\+flower\+at\+the\+Wall$/);
});

test("opens and closes the filter disclosure from the keyboard", async ({
  page
}) => {
  await page.route("**/api/catalog", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        books: [
          {
            book_id: "agot",
            book_title: "A Game of Thrones",
            book_sequence: 1,
            povs: ["ARYA", "EDDARD"]
          }
        ]
      })
    });
  });

  await page.goto("/");
  const trigger = page.getByRole("button", { name: /Books & POVs/ });
  const collapse = page.locator(".filter-collapse");

  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(collapse).toHaveAttribute("inert", "");

  await trigger.focus();
  await page.keyboard.press("Enter");
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await expect(collapse).not.toHaveAttribute("inert", "");
  await expect(
    page.getByRole("radio", { name: "A Game of Thrones" })
  ).toBeVisible();

  await page.keyboard.press("Space");
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(collapse).toHaveAttribute("inert", "");
});
