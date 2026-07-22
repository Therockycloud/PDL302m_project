#!/usr/bin/env node

import { readFileSync, statSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const deckDir = dirname(fileURLToPath(import.meta.url));
const requiredFiles = ['index.html', 'OUTLINE.md', 'SCRIPT.md'];

function stripHtmlComments(html) {
  return html.replace(/<!--[\s\S]*?-->/g, '');
}

function slideOpeningTags(html) {
  return html.match(/<section\b[^>]*>/g) ?? [];
}

function isSlideTag(tag) {
  const classAttribute = tag.match(/(?:^|\s)class\s*=\s*(["'])([^"']*)\1/);
  return classAttribute?.[2].split(/\s+/).includes('slide') ?? false;
}

function layoutAttributes(tag) {
  return [...tag.matchAll(/(?:^|\s)data-layout\s*=\s*(["'])(S\d{2})\1/g)];
}

function registeredLayoutValue(tag) {
  return tag.match(/(?:^|\s)data-layout\s*=\s*(["'])(S\d{2}|SWISS-COVER-ASCII|SWISS-CLOSING-ASCII)\1/)?.[2] ?? null;
}

function slideCountFailure(slides) {
  return slides.length === 35 ? null : `Expected 35 slide sections, found ${slides.length}`;
}

function numberedValues(content, expression) {
  return [...content.matchAll(expression)].map((match) => match[1]);
}

function sequenceFailure(label, values) {
  if (values.length !== 35) return `Expected 35 ${label}, found ${values.length}`;
  for (let index = 0; index < 35; index += 1) {
    const expected = String(index + 1).padStart(2, '0');
    if (values[index] !== expected) {
      return `Expected ${label} sequence 01..35; position ${index + 1} is ${values[index] ?? 'missing'}`;
    }
  }
  return null;
}

function staysWithin(basePath, candidatePath) {
  const pathFromBase = relative(basePath, candidatePath);
  return pathFromBase !== '..'
    && !pathFromBase.startsWith(`..${sep}`)
    && !isAbsolute(pathFromBase);
}

function hasTraversalSegment(path) {
  return path.split(/[\\/]/).includes('..');
}

function isRegularFile(path) {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

function runSelfTests() {
  const fixture = [
    '<!-- data-layout="S99" must not count -->',
    '<section class="slide">',
    '<section class="slide" data-layout="S01">',
    '<section class="not-a-slide" data-layout="S99">',
  ].join('');
  const slides = slideOpeningTags(fixture).filter(isSlideTag);
  if (slides.length !== 2 || layoutAttributes(slides[0]).length !== 0 || layoutAttributes(slides[1]).length !== 1) {
    throw new Error('slide-scoped layout self-test failed');
  }
  const commentedSlides = '<section class="slide">'.repeat(33);
  const falsePassFixture = [
    '<section data-class="slide" x-data-layout="S01">',
    '<section class="slide">',
    '<section class="slide" data-layout="S01">',
    `<!-- ${commentedSlides} -->`,
  ].join('');
  const falsePassSlides = slideOpeningTags(stripHtmlComments(falsePassFixture)).filter(isSlideTag);
  if (falsePassSlides.length !== 2 || !slideCountFailure(falsePassSlides)
    || layoutAttributes(falsePassSlides[0]).length !== 0 || layoutAttributes(falsePassSlides[1]).length !== 1) {
    throw new Error('commented or prefixed attribute self-test failed');
  }
  const prefixedLayoutSlide = '<section class="slide" x-data-layout="S01">';
  if (layoutAttributes(slideOpeningTags(prefixedLayoutSlide)[0]).length !== 0) {
    throw new Error('prefixed data-layout attribute self-test failed');
  }
  if (registeredLayoutValue('<section class="slide" data-layout="SWISS-COVER-ASCII">') !== 'SWISS-COVER-ASCII'
    || registeredLayoutValue('<section class="slide" data-layout="SWISS-CLOSING-ASCII">') !== 'SWISS-CLOSING-ASCII') {
    throw new Error('registered cover/closing layout self-test failed');
  }
  if (sequenceFailure('outline rows', ['01', '03']) === null) {
    throw new Error('number-sequence self-test failed');
  }
  const duplicateSequence = Array.from({ length: 35 }, (_, index) => String(index + 1).padStart(2, '0'));
  duplicateSequence[17] = '17';
  if (sequenceFailure('outline rows', duplicateSequence) === null
    || sequenceFailure('SCRIPT headings', duplicateSequence) === null) {
    throw new Error('full-length duplicate sequence self-test failed');
  }
  const image = 'images/nested evidence/missing%20asset.png';
  if (!slideOpeningTags(`<section class="slide" data-image="${image}">`).length
    || !(`<img src="${image}">`.match(/images\/[^"'()]+/g) ?? []).includes(image)) {
    throw new Error('nested image-reference self-test failed');
  }
  if (!staysWithin(deckDir, resolve(deckDir, 'images/nested/file.png'))
    || staysWithin(deckDir, resolve(deckDir, '../outside.png'))
    || !hasTraversalSegment('images/../outside.png')) {
    throw new Error('path-containment self-test failed');
  }
  if (!isRegularFile(fileURLToPath(import.meta.url)) || isRegularFile(resolve(deckDir, 'missing-file.png'))) {
    throw new Error('regular-file self-test failed');
  }
  console.log('Swiss deck validator self-tests passed.');
}

function validateDeck() {
  const contents = new Map();
  const failures = [];

  for (const file of requiredFiles) {
    const path = resolve(deckDir, file);
    try {
      if (!statSync(path).isFile()) {
        failures.push(`Required artifact is not a regular file: ${file}`);
        continue;
      }
      contents.set(file, readFileSync(path, 'utf8'));
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      failures.push(`Unable to read required artifact ${file}: ${detail}`);
    }
  }

  const html = contents.get('index.html') ?? '';
  const outline = contents.get('OUTLINE.md') ?? '';
  const script = contents.get('SCRIPT.md') ?? '';
  const parsedHtml = stripHtmlComments(html);
  const slides = slideOpeningTags(parsedHtml).filter(isSlideTag);
  const slideFailure = slideCountFailure(slides);
  if (slideFailure) failures.push(slideFailure);

  const substantiveLayouts = [];
  for (let index = 0; index < slides.length; index += 1) {
    const layouts = layoutAttributes(slides[index]);
    const slideNumber = index + 1;
    if (slideNumber === 1) {
      if (registeredLayoutValue(slides[index]) !== 'SWISS-COVER-ASCII') {
        failures.push('Slide 01 must use data-layout="SWISS-COVER-ASCII"');
      }
      continue;
    }
    if (slideNumber === 35) {
      if (registeredLayoutValue(slides[index]) !== 'SWISS-CLOSING-ASCII') {
        failures.push('Slide 35 must use data-layout="SWISS-CLOSING-ASCII"');
      }
      continue;
    }
    if (layouts.length !== 1) {
      failures.push(`Slide ${String(slideNumber).padStart(2, '0')} must have exactly one data-layout="Sxx" attribute, found ${layouts.length}`);
      continue;
    }
    substantiveLayouts.push(layouts[0]);
  }
  if (substantiveLayouts.length !== 33) {
    failures.push(`Expected 33 substantive data-layout attributes, found ${substantiveLayouts.length}`);
  }

  const outlineFailure = sequenceFailure('outline rows', numberedValues(outline, /^\| (\d{2}) \|/gm));
  if (outlineFailure) failures.push(outlineFailure);
  const scriptFailure = sequenceFailure('SCRIPT headings', numberedValues(script, /^## Slide (\d{2}):/gm));
  if (scriptFailure) failures.push(scriptFailure);

  for (const [file, content] of contents) {
    const placeholders = content.match(/\[必填\]|TBD|TODO/g) ?? [];
    if (placeholders.length > 0) failures.push(`Found placeholder(s) in ${file}: ${[...new Set(placeholders)].join(', ')}`);
  }

  const imageReferences = [...new Set(parsedHtml.match(/images\/[^"'()]+/g) ?? [])];
  for (const imageReference of imageReferences) {
    let decodedReference;
    try {
      decodedReference = decodeURIComponent(imageReference);
    } catch {
      failures.push(`Invalid URL-encoded local image: ${imageReference}`);
      continue;
    }
    const assetPath = resolve(deckDir, decodedReference);
    if (hasTraversalSegment(decodedReference) || !staysWithin(deckDir, assetPath)) {
      failures.push(`Local image path escapes deck directory: ${imageReference}`);
      continue;
    }
    try {
      if (!statSync(assetPath).isFile()) failures.push(`Local image is not a regular file: ${imageReference}`);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      failures.push(`Missing local image: ${imageReference} (${detail})`);
    }
  }

  if (failures.length > 0) {
    console.error('Swiss deck validation failed:');
    for (const failure of failures) console.error(`- ${failure}`);
    process.exitCode = 1;
  } else {
    console.log(`Swiss deck validation passed (${slides.length} slides, ${imageReferences.length} local images).`);
  }
}

if (process.argv.includes('--self-test')) runSelfTests();
else validateDeck();
