import { expect, test, type Page } from "@playwright/test";

async function login(page: Page, password = process.env.ADMIN_PASSWORD ?? "") {
  await page.goto("/login");
  const passwordInput = page.locator('input[name="password"]');
  await expect(passwordInput).toBeVisible();
  // Clear any browser autofill, then set password (empty is valid when unset).
  await passwordInput.fill("");
  if (password) {
    await passwordInput.fill(password);
  }
  await page.getByRole("button", { name: /sign in/i }).click();
  await expect(page).toHaveURL(/\/dashboard/, { timeout: 20_000 });
  // Page title is in topbar h1 when present; fall back to dashboard content.
  const h1 = page.locator("h1");
  if ((await h1.count()) > 0) {
    await expect(h1.first()).toContainText(/dashboard/i);
  } else {
    await expect(page.locator("#dashboard-panel, .ox-content, main").first()).toBeVisible();
  }
}

/** Accept the next native confirm/alert once. */
function acceptNextDialog(page: Page) {
  return new Promise<void>((resolve) => {
    page.once("dialog", async (d) => {
      await d.accept();
      resolve();
    });
  });
}

test.describe("oxllm admin MVP", () => {
  test("login and navigate sections", async ({ page }) => {
    await login(page);
    for (const [path, title] of [
      ["/keys", /api keys/i],
      ["/teams", /teams/i],
      ["/users", /users/i],
      ["/spend", /spend/i],
      ["/models", /models/i],
      ["/dashboard", /dashboard/i],
    ] as const) {
      await page.goto(path);
      await expect(page.locator("h1")).toContainText(title);
    }
  });

  test("create team, user, and key", async ({ page }) => {
    await login(page);
    const stamp = Date.now().toString(36);

    await page.goto("/teams");
    await page.getByRole("button", { name: /create team/i }).click();
    await page.locator('#team_name').fill(`e2e-team-${stamp}`);
    await page.locator('#team_max_budget').fill("25");
    await page.getByRole("button", { name: /^create team$/i }).click();
    await expect(page.locator("#teams-panel")).toContainText(`e2e-team-${stamp}`, {
      timeout: 20_000,
    });
    await expect(page.locator("#teams-panel")).toContainText(/team-/i);

    await page.goto("/users");
    await page.getByRole("button", { name: /create user/i }).click();
    await page.locator('#user_name').fill(`e2e-user-${stamp}`);
    await page.locator('#user_max_budget').fill("10");
    await page.getByRole("button", { name: /^create user$/i }).click();
    await expect(page.locator("#users-panel")).toContainText(`e2e-user-${stamp}`, {
      timeout: 20_000,
    });
    await expect(page.locator("#users-panel")).toContainText(/user-/i);

    await page.goto("/keys");
    await page.getByRole("button", { name: /create key/i }).click();
    await page.locator('#key_alias').fill(`e2e-key-${stamp}`);
    // Team is a <select> of existing teams
    const teamSelect = page.locator("select#team_id");
    await expect(teamSelect).toBeVisible();
    const teamOptions = await teamSelect.locator("option").count();
    if (teamOptions > 1) {
      // pick first real team (index 0 is "(none)")
      await teamSelect.selectOption({ index: 1 });
    }
    await page.getByRole("button", { name: /generate key/i }).click();
    await expect(page.locator("#key-result")).toContainText(/key created/i, {
      timeout: 20_000,
    });
    await expect(page.locator("#new-key-value")).toContainText(/sk-ox-/);
    await page.getByRole("button", { name: /^done$/i }).click();
    await expect(page.locator("#keys-panel")).toContainText(`e2e-key-${stamp}`, {
      timeout: 20_000,
    });
  });

  test("disable key marks row Disabled", async ({ page }) => {
    await login(page);
    const stamp = Date.now().toString(36);
    const alias = `del-key-${stamp}`;

    await page.goto("/keys");
    await page.getByRole("button", { name: /create key/i }).click();
    await page.locator('#key_alias').fill(alias);
    await page.getByRole("button", { name: /generate key/i }).click();
    await expect(page.locator("#new-key-id")).toBeVisible({ timeout: 20_000 });
    const keyId = (await page.locator("#new-key-id").innerText()).trim();
    expect(keyId).toMatch(/^key-/);

    await page.getByRole("button", { name: /^done$/i }).click();
    await expect(page.locator("#keys-panel")).toContainText(alias, { timeout: 20_000 });

    const row = page.locator(`#key-row-${keyId}`);
    await expect(row).toBeVisible();
    await expect(row.getByText(/active/i)).toBeVisible();

    const dialogDone = acceptNextDialog(page);
    await row.getByRole("button", { name: /disable/i }).click();
    await dialogDone;

    await expect(row.getByText(/disabled/i)).toBeVisible({ timeout: 15_000 });
    await expect(row.getByRole("button", { name: /disable/i })).toHaveCount(0);
    await expect(row.getByRole("button", { name: /enable/i })).toBeVisible();
    await expect(row.getByRole("button", { name: /remove/i })).toBeVisible();

    const enableDone = acceptNextDialog(page);
    await row.getByRole("button", { name: /enable/i }).click();
    await enableDone;
    await expect(row.getByText(/active/i)).toBeVisible({ timeout: 15_000 });
    await expect(row.getByRole("button", { name: /disable/i })).toBeVisible();

    const disableAgain = acceptNextDialog(page);
    await row.getByRole("button", { name: /disable/i }).click();
    await disableAgain;
    await expect(row.getByText(/disabled/i)).toBeVisible({ timeout: 15_000 });

    const removeDone = acceptNextDialog(page);
    await row.getByRole("button", { name: /remove/i }).click();
    await removeDone;
    await expect(page.locator(`#key-row-${keyId}`)).toHaveCount(0, { timeout: 15_000 });
  });

  test("spend page refresh does not duplicate header", async ({ page }) => {
    await login(page);
    await page.goto("/spend");
    await expect(page.locator("#spend-panel")).toBeVisible();

    const countBefore = await page.locator("#spend-panel h3", { hasText: "Spend Logs" }).count();
    expect(countBefore).toBe(1);

    // Wait for an observable post-refresh signal: the HTMX outerHTML swap of
    // #spend-panel. A fixed sleep can pass before the swap lands on a slow
    // stack, silently skipping the duplicate-header window.
    await page.evaluate(() => {
      (window as unknown as Record<string, unknown>).__spendSwapped = false;
      document.body.addEventListener("htmx:afterSwap", (event: Event) => {
        const detail = (event as CustomEvent).detail as { target?: HTMLElement } | undefined;
        if (detail?.target?.id === "spend-panel") {
          (window as unknown as Record<string, unknown>).__spendSwapped = true;
        }
      });
    });
    await page.getByRole("button", { name: /refresh/i }).click();
    await page.waitForFunction(
      () => (window as unknown as Record<string, unknown>).__spendSwapped === true,
      undefined,
      { timeout: 15_000 },
    );
    await expect
      .poll(async () => page.locator("#spend-panel h3", { hasText: "Spend Logs" }).count())
      .toBe(1);

    const countAfter = await page.locator("#spend-panel h3", { hasText: "Spend Logs" }).count();
    expect(countAfter).toBe(1);
    await expect(page.locator("#spend-panel")).toHaveCount(1);
  });

  test("spend table renders cost column values when present", async ({ page }) => {
    await login(page);
    await page.goto("/spend");
    await expect(page.locator("#spend-panel table")).toBeVisible();
    const body = await page.locator("#spend-rows").innerText();
    expect(body.toLowerCase()).not.toContain("error");
    expect(body.toLowerCase()).not.toContain("failed");
    const hasDollar = body.includes("$");
    const hasDash = body.includes("—") || body.includes("-");
    expect(hasDollar || hasDash || /no spend/i.test(body)).toBeTruthy();
  });

  test("spend filter by model keeps filter value and updates panel", async ({ page }) => {
    await login(page);
    await page.goto("/spend");
    await page.locator("#filter_model").fill("text-embedding-3-small");
    await page.getByRole("button", { name: /^filter$/i }).click();
    // After HTMX swap the panel re-renders with filters preserved
    await expect(page.locator("#filter_model")).toHaveValue("text-embedding-3-small", {
      timeout: 15_000,
    });
    await expect(page.locator("#spend-panel")).toBeVisible();
    await expect(page.locator(".ox-flash-error")).toHaveCount(0);
    const rows = await page.locator("#spend-rows").innerText();
    // Matching rows, empty state, or only embedding models — no crash
    expect(rows.toLowerCase()).not.toContain("internal server");
  });

  test("spend sort headers and date-range export csv work", async ({ page }) => {
    await login(page);
    await page.goto("/spend");
    await expect(page.locator("#spend-panel")).toBeVisible();

    // Default Start/End dates are always present for export
    const start = page.locator("#filter_start");
    const end = page.locator("#filter_end");
    await expect(start).not.toHaveValue("");
    await expect(end).not.toHaveValue("");
    const startVal = await start.inputValue();
    const endVal = await end.inputValue();
    expect(startVal).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(endVal).toMatch(/^\d{4}-\d{2}-\d{2}$/);

    // Sort by model (sortable header link)
    const modelSort = page.locator("a.ox-sort-link", { hasText: /^Model/ });
    await expect(modelSort).toBeVisible();
    await modelSort.click();
    await expect(page.locator("#spend-panel")).toBeVisible();
    await expect(page.locator(".ox-flash-error")).toHaveCount(0);

    // Export CSV for the UI date range
    const exportBtn = page.getByRole("button", { name: /export csv/i });
    await expect(exportBtn).toBeVisible();
    const downloadPromise = page.waitForEvent("download", { timeout: 15_000 });
    await exportBtn.click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/\.csv$/i);
    expect(download.suggestedFilename()).toContain(startVal);
    expect(download.suggestedFilename()).toContain(endVal);
  });

  test("dashboard shows health and activity bars with non-zero width when spend exists", async ({
    page,
  }) => {
    await login(page);
    await page.goto("/dashboard");
    await expect(page.locator("#dashboard-panel")).toBeVisible();
    await expect(page.locator(".ox-stat-card").first()).toBeVisible();

    const onlineOrDegraded = page.locator(".ox-badge", { hasText: /Online|Degraded/i });
    await expect(onlineOrDegraded.first()).toBeVisible();

    const fills = page.locator(".ox-activity-fill");
    const n = await fills.count();
    if (n > 0) {
      const widths = await fills.evaluateAll((els) =>
        els.map((el) => (el as HTMLElement).style.width),
      );
      const anyPositive = widths.some((w) => {
        const n = parseFloat(w);
        return !Number.isNaN(n) && n > 0;
      });
      expect(anyPositive).toBeTruthy();
    }
  });

  test("models page lists configured models", async ({ page }) => {
    await login(page);
    await page.goto("/models");
    await expect(page.locator("h1")).toContainText(/models/i);
    await expect(page.getByText(/configured models/i)).toBeVisible();
    const table = page.locator(".ox-table");
    await expect(table).toBeVisible();
    const text = await table.innerText();
    expect(/gpt-4o|grok|embedding/i.test(text)).toBeTruthy();
  });

  test("theme toggle flips data-theme", async ({ page }) => {
    await login(page);
    await page.goto("/dashboard");
    const html = page.locator("html");
    const before = await html.getAttribute("data-theme");
    await page.locator(".ox-theme-btn").click();
    const after = await html.getAttribute("data-theme");
    expect(after).not.toEqual(before);
    expect(["light", "dark"]).toContain(after ?? "");
  });

  test("logout returns to login", async ({ page }) => {
    await login(page);
    const dialogDone = acceptNextDialog(page);
    await page.getByRole("link", { name: /logout/i }).click();
    await dialogDone;
    await expect(page).toHaveURL(/\/login/);
    await expect(page.getByRole("button", { name: /sign in/i })).toBeVisible();
  });

  test("csrf meta is present and posts succeed", async ({ page }) => {
    await login(page);
    const csrf = await page.locator('meta[name="csrf-token"]').getAttribute("content");
    expect(csrf && csrf.length > 10).toBeTruthy();

    const stamp = Date.now().toString(36);
    await page.goto("/teams");
    await page.getByRole("button", { name: /create team/i }).click();
    await page.locator('#team_name').fill(`csrf-team-${stamp}`);
    await page.getByRole("button", { name: /^create team$/i }).click();
    await expect(page.locator("#teams-panel")).toContainText(`csrf-team-${stamp}`, {
      timeout: 20_000,
    });
  });

  test("edit and delete team", async ({ page }) => {
    await login(page);
    const stamp = Date.now().toString(36);
    const name = `edit-team-${stamp}`;
    await page.goto("/teams");
    await page.getByRole("button", { name: /create team/i }).click();
    await page.locator('#team_name').fill(name);
    await page.getByRole("button", { name: /^create team$/i }).click();
    await expect(page.locator("#teams-panel")).toContainText(name, { timeout: 20_000 });

    const row = page.locator("tr", { hasText: name }).first();
    await row.getByRole("button", { name: /^edit$/i }).click();
    const form = page.locator("tr.ox-edit-row form").first();
    await expect(form).toBeVisible();
    await form.locator('input[name="team_name"]').fill(`${name}-renamed`);
    await form.locator('input[name="max_budget"]').fill("99");
    await form.getByRole("button", { name: /^save$/i }).click();
    await expect(page.locator("#teams-panel")).toContainText(`${name}-renamed`, {
      timeout: 20_000,
    });
    await expect(page.locator("#teams-panel")).toContainText("99");

    const dialogDone = acceptNextDialog(page);
    await page
      .locator("tr", { hasText: `${name}-renamed` })
      .first()
      .getByRole("button", { name: /^delete$/i })
      .click();
    await dialogDone;
    await expect(page.locator("#teams-panel")).not.toContainText(`${name}-renamed`, {
      timeout: 20_000,
    });
  });

  test("edit and delete user", async ({ page }) => {
    await login(page);
    const stamp = Date.now().toString(36);
    const name = `edit-user-${stamp}`;
    await page.goto("/users");
    await page.getByRole("button", { name: /create user/i }).click();
    await page.locator('#user_name').fill(name);
    await page.getByRole("button", { name: /^create user$/i }).click();
    await expect(page.locator("#users-panel")).toContainText(name, { timeout: 20_000 });

    const row = page.locator("tr", { hasText: name }).first();
    await row.getByRole("button", { name: /^edit$/i }).click();
    const form = page.locator("tr.ox-edit-row form").first();
    await expect(form).toBeVisible();
    await form.locator('input[name="user_name"]').fill(`${name}-renamed`);
    await form.locator('input[name="max_budget"]').fill("12");
    await form.getByRole("button", { name: /^save$/i }).click();
    await expect(page.locator("#users-panel")).toContainText(`${name}-renamed`, {
      timeout: 20_000,
    });

    const dialogDone = acceptNextDialog(page);
    await page
      .locator("tr", { hasText: `${name}-renamed` })
      .first()
      .getByRole("button", { name: /^delete$/i })
      .click();
    await dialogDone;
    await expect(page.locator("#users-panel")).not.toContainText(`${name}-renamed`, {
      timeout: 20_000,
    });
  });

  test("key rotate shows new secret once", async ({ page }) => {
    await login(page);
    const stamp = Date.now().toString(36);
    await page.goto("/keys");
    await page.getByRole("button", { name: /create key/i }).click();
    await page.locator('#key_alias').fill(`rot-${stamp}`);
    await page.getByRole("button", { name: /generate key/i }).click();
    await expect(page.locator("#new-key-id")).toBeVisible({ timeout: 20_000 });
    const keyId = (await page.locator("#new-key-id").innerText()).trim();
    await page.getByRole("button", { name: /^done$/i }).click();
    await expect(page.locator("#keys-panel")).toContainText(`rot-${stamp}`, { timeout: 20_000 });

    const row = page.locator(`#key-row-${keyId}`);
    const dialogDone = acceptNextDialog(page);
    await row.getByRole("button", { name: /rotate/i }).click();
    await dialogDone;
    await expect(page.locator("#key-result")).toContainText(/key rotated|key created/i, {
      timeout: 20_000,
    });
    await expect(page.locator("#new-key-value")).toContainText(/sk-ox-/);
  });

  test("cost calculator returns estimate", async ({ page }) => {
    await login(page);
    await page.goto("/calculator");
    await expect(page.locator("h1")).toContainText(/calculator/i);
    await page.locator('input[name="model"]').fill("gpt-4o");
    await page.locator('input[name="prompt_tokens"]').fill("1000");
    await page.locator('input[name="completion_tokens"]').fill("500");
    await page.getByRole("button", { name: /^calculate$/i }).click();
    await expect(page.locator("#calc-result")).toContainText(/\$|estimated|price/i, {
      timeout: 15_000,
    });
  });

  test("spend date presets 7d and 30d", async ({ page }) => {
    await login(page);
    await page.goto("/spend");
    await expect(page.locator("#filter_start")).toBeVisible();
    await page.getByRole("button", { name: /^7d$/i }).click();
    await expect(page.locator("#spend-panel")).toBeVisible({ timeout: 15_000 });
    const start7 = await page.locator("#filter_start").inputValue();
    const end7 = await page.locator("#filter_end").inputValue();
    expect(start7).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(end7).toMatch(/^\d{4}-\d{2}-\d{2}$/);

    await page.getByRole("button", { name: /^30d$/i }).click();
    await expect(page.locator("#spend-panel")).toBeVisible({ timeout: 15_000 });
    const start30 = await page.locator("#filter_start").inputValue();
    const end30 = await page.locator("#filter_end").inputValue();
    // A 30d preset must start no later than the 7d preset; comparing
    // `start30 < end30` here would be tautological.
    expect(start30 <= start7).toBeTruthy();
    expect(end30).toBe(end7);
  });
});
