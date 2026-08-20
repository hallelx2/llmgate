# Contributing to llmgate

## Commits and PR titles

[Conventional Commits](https://www.conventionalcommits.org/). Types:
`feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `build`,
`ci`. Scope is the package. The body explains *why*, not what the diff
already shows.

### A breaking change must carry its `!` in the PR title

This repository squash-merges. **Squash-merge takes the PR title as the
commit subject and discards every per-commit subject**, so a `!` that
exists only on a commit inside the branch does not survive the merge.

That already happened once. PR #9 was authored as `feat(pricing)!:` —
it changed the public `Usage` and `Price` structs — but the PR title had
no `!`, so `main` reads:

```
af5a9ec feat(pricing): tiered cost accounting, model-ID normalization, and live price feeds (#9)
```

The marker survives only inside the squashed body, where no tooling and
no casual reader will look. The release after it was nearly cut as a
patch.

So: if any commit on your branch is `type(scope)!:`, **the PR title must
be `type(scope)!:` too**, and the PR body must carry a
`BREAKING CHANGE:` paragraph describing the migration.

## Versioning

Semantic versioning, pre-1.0: **a breaking change bumps the minor**
(`v0.3.0` → `v0.4.0`), not the patch. Adding a field to an exported
struct is breaking here, because positional struct literals stop
compiling even though named-field callers do not.

Every release needs a matching `## [x.y.z]` section in
[`CHANGELOG.md`](CHANGELOG.md) before the tag is pushed. The release
workflow enforces this — a tag whose version has no changelog section
fails the release rather than shipping undocumented.

## Before you open a PR

```bash
gofmt -l .                  # must be empty
go vet ./...
go build ./...
go test -race -count=1 ./...
golangci-lint run           # config in .golangci.yml
```

Tests that register a global pricing override must unregister it. Those
overrides are process-global, so a test that leaves one behind makes an
unrelated test fail under `-shuffle=on` and only then.

## Releasing

1. Land everything for the release on `main`.
2. Add the `## [x.y.z]` section to `CHANGELOG.md`, with breaking changes
   in their own section ahead of the fixes.
3. Tag and push: `git tag vx.y.z && git push origin vx.y.z`.

The `Release` workflow validates the tag is semver, checks the changelog
section exists, verifies `go mod tidy` is clean, runs vet/build/race
tests, publishes the GitHub Release, and warms `proxy.golang.org` so
`pkg.go.dev` indexes the version immediately instead of waiting for its
next scan.

A Go module tag cannot be moved once the proxy has served it. Verify
before tagging, not after.
