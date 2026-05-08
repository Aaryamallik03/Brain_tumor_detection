---
name: Lighthouse Best Practice Analyzer
---

## Purpose

Automatically scan repository for console errors using Lighthouse audits, categorize issues, and either commit fixes directly to PRs or create GitHub issues for complex problems requiring manual intervention.

## Execution Steps

### 1. Initial Setup & Lighthouse Audit

- Identify the target environment (staging/preview URL for PR, production for main branch)
- Run Lighthouse audit with best practices category enabled
- Focus specifically on console error audits:
    - `errors-in-console` audit results
    - `no-console-logs` warnings (development artifacts)
    - Browser console errors during page load

### 2. Error Collection & Analysis

- Extract all console errors from Lighthouse report
- Collect error details:
    - Error message and stack trace
    - Source file and line number
    - Error type (TypeError, ReferenceError, NetworkError, etc.)
    - Context (component/page where error occurred)
- Group related errors (same root cause)

### 3. Error Triage & Classification

### Category A: Auto-Fixable Issues

- **Leftover console.log/debug statements** → Remove automatically
- **Undefined variable references** → Add null checks
- **Missing error boundaries** → Add try-catch or error boundary
- **Simple typos in code** → Fix directly

### Category B: Requires Investigation

- **Third-party library errors** → Create issue with details
- **API/Network failures** → Document and create issue
- **Complex logic errors** → Create issue with reproduction steps
- **Build/bundling issues** → Create issue with stack trace

### 4. Action Execution

**For Category A (Auto-Fix):**

1. Create a branch if not already on PR branch: `fix/lighthouse-console-errors-{date}`
2. Apply fixes to affected files
3. Add HTML marker comments for tracking
4. Commit with structured message
5. Update or create PR with checklist

**For Category B (Investigation Required):**

1. Create GitHub issue with detailed template
2. Add labels: `lighthouse`, `console-error`, `needs-investigation`
3. Link to Lighthouse report and error context

## File Modification Patterns

### Adding Error Boundaries

```jsx
// Before
export default function MyComponent() {
  return <div>{data.value}</div>
}

// After
<!-- lighthouse-error-boundary -->
export default function MyComponent() {
  try {
    if (!data?.value) return null
    return <div>{data.value}</div>
  } catch (error) {
    console.error('MyComponent error:', error)
    return <div>Error loading content</div>
  }
}
<!-- /lighthouse-error-boundary -->

```

### Removing Console Statements

```jsx
// Before
console.log('Debug info:', someData)
console.debug('Testing:', value)

// After - removed by lighthouse-console-analyzer
<!-- lighthouse-console-cleaned -->

```

## Commit Message Format

```
fix(lighthouse): resolve console errors from audit

Auto-fixed console errors identified by Lighthouse best practices audit:

Category A Fixes:
- Removed 3 console.log statements in components/Dashboard.tsx
- Added null check for undefined variable in utils/api.ts
- Added error boundary to components/UserProfile.tsx

Lighthouse Score Impact:
- Before: Best Practices Score XX/100
- After: Best Practices Score YY/100
- Console errors reduced from N to M

Issues Created:
- #123: Third-party library error in analytics script
- #124: Network timeout in API endpoint

<!-- lighthouse-console-analyzer: run-{timestamp} -->

```

## GitHub Issue Template (Category B)

```markdown
---
title: "Console Error: {Error Type} in {Component/File}"
labels: lighthouse, console-error, needs-investigation
---
## Error Details

**Detected by:** Lighthouse Best Practices Audit
**Audit Date:** {timestamp}
**Environment:** {staging/production URL}
**Lighthouse Score Impact:** -{X} points

### Error Message

```

{full error message}

```

### Stack Trace

```

{stack trace if available}

```

### Location
- **File:** `{file path}`
- **Line:** {line number}
- **Component/Context:** {component name or page}

### Error Classification
**Category:** {error type - e.g., Third-party library, Network, Logic}
**Severity:** {High/Medium/Low based on frequency and impact}
**Frequency:** {how often error occurs}

### Reproduction Steps
1. Navigate to {URL}
2. Open browser console
3. {specific actions that trigger error}
4. Observe error in console

### Investigation Needed
- [ ] Identify root cause
- [ ] Determine if error affects functionality
- [ ] Check if error occurs in production
- [ ] Test potential fixes
- [ ] Verify fix doesn't break other functionality

### Lighthouse Report
**Full Report:** {link to Lighthouse report if saved}
**Related Audits:**
- errors-in-console: {pass/fail}
- {other relevant audits}

### Additional Context
{any other relevant information from Lighthouse or manual inspection}

<!-- lighthouse-issue-tracker: {issue-id} -->

```

## PR Comment Template (When Committing Fixes)

```markdown
## 🔦 Lighthouse Console Error Analysis

**Audit Run:** {timestamp}
**Branch:** `{branch-name}`
**Environment Tested:** {URL}

### ✅ Auto-Fixed Issues (Category A)

| File | Issue | Fix Applied |
|------|-------|-------------|
| `{file-path}` | {issue description} | {fix description} |

**Total Fixes:** {N} issues resolved automatically

### 📋 Issues Created for Investigation (Category B)

| Issue | Type | Severity | Link |
|-------|------|----------|------|
| #{issue-number} | {error type} | {severity} | [View Issue]({url}) |

**Total Issues:** {M} issues require manual investigation

### 📊 Lighthouse Score Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Best Practices | {XX}/100 | {YY}/100 | +{diff} |
| Console Errors | {N} errors | {M} errors | -{fixed} |

### 🔍 Error Summary

**Types Fixed:**
- Console.log statements removed: {count}
- Null checks added: {count}
- Error boundaries added: {count}

**Types Requiring Investigation:**
- Third-party errors: {count}
- Network errors: {count}
- Logic errors: {count}

### ✓ Verification Checklist
- [x] Lighthouse audit completed
- [x] Auto-fixable errors resolved
- [x] Complex errors documented in issues
- [x] Code changes reviewed and tested
- [ ] Verify fixes don't introduce regressions

---
*Generated by Lighthouse Console Error Analyzer*
<!-- lighthouse-pr-analysis: {run-id} -->

```

## Detection & Update Logic

### HTML Markers for Tracking

```html
<!-- lighthouse-error-boundary -->
{error handling code}
<!-- /lighthouse-error-boundary -->

<!-- lighthouse-console-cleaned -->
<!-- removed console.log at line {X} on {date} -->

<!-- lighthouse-issue-tracker: {issue-id} -->

<!-- lighthouse-pr-analysis: {run-id} -->

```

### Update Existing Markers

When re-running analysis:

1. Check for existing `<!-- lighthouse-pr-analysis -->` markers in PR comments
2. Update with new results rather than creating duplicate comments
3. Track resolution status of previously created issues

## Integration Requirements

### Tools Needed

- **Lighthouse CLI** or **PageSpeed Insights API** for audits
- **GitHub API** for:
    - Creating issues
    - Commenting on PRs
    - Committing fixes to branches
- **File system access** for reading/modifying source files

### Configuration

```json
{
  "lighthouse": {
    "categories": ["best-practices"],
    "audits": ["errors-in-console", "no-console-logs"],
    "throttling": "mobile3G",
    "output": "json"
  },
  "autofix": {
    "enabled": true,
    "categories": ["console-logs", "null-checks", "error-boundaries"],
    "maxChangesPerRun": 10
  },
  "issueCreation": {
    "enabled": true,
    "labels": ["lighthouse", "console-error", "needs-investigation"],
    "assignees": []
  }
}

```

## Workflow Decision Tree

```
Run Lighthouse Audit
    ↓
Collect Console Errors
    ↓
For Each Error:
    ↓
    ├─→ Simple/Auto-fixable? (Category A)
    │       ↓
    │   Apply fix → Commit to PR
    │       ↓
    │   Update PR comment
    │
    └─→ Complex/Needs Investigation? (Category B)
            ↓
        Create GitHub Issue
            ↓
        Link in PR comment

```

## Success Criteria

- ✅ Lighthouse audit runs successfully
- ✅ All console errors identified and categorized
- ✅ Auto-fixable issues committed with clear messages
- ✅ Complex issues documented in GitHub issues with full context
- ✅ PR updated with comprehensive analysis
- ✅ Best Practices score improved
- ✅ No regressions introduced by auto-fixes

## Example Scenarios

### Scenario 1: PR with Console Logs

- **Detected:** 5 console.log statements in dev code
- **Action:** Auto-remove all 5, commit with lighthouse message
- **Result:** Clean PR, improved Lighthouse score

### Scenario 2: Third-Party Script Error

- **Detected:** Error from analytics.js library
- **Action:** Create issue #456 with full error details
- **Result:** Team investigates vendor integration

### Scenario 3: Mixed Issues

- **Detected:** 3 console.logs + 2 undefined variable errors + 1 network error
- **Action:**
    - Auto-fix: Remove logs, add null checks
    - Create Issue: Document network error
    - Update PR: Show both actions in summary