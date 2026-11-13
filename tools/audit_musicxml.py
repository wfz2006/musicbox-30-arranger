"""Read-only MusicXML audit for the user's exact 30-note instrument.

Python 3.10+, standard library only. Exit 0 means the supported, linear
score passes the mechanical audit, NOT that its musical arrangement is good.
Navigation, grace timing, transposition doubling and ottavas are fail-closed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ALLOWED = frozenset([53, 55, 60, 62, 64, 65, 67, 69, 70, 71, 72, 73, 74,
                     75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                     88, 89, 91, 93])
STEPS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
UNITS = {'maxima': 32, 'long': 16, 'breve': 8, 'whole': 4, 'half': 2,
         'quarter': 1, 'eighth': F(1, 2), '16th': F(1, 4), '32nd': F(1, 8),
         '64th': F(1, 16), '128th': F(1, 32), '256th': F(1, 64)}
PITCH_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


def pitch_name(midi):
    if F(midi).denominator != 1:
        return f'MIDI:{midi}'
    midi = int(midi)
    return f'{PITCH_NAMES[midi % 12]}{midi // 12 - 1}'


def serial(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [serial(v) for v in value]
    return value


def read_xml(path):
    path = Path(path)
    if path.suffix.lower() == '.mxl':
        with zipfile.ZipFile(path) as z:
            container = ET.fromstring(z.read('META-INF/container.xml'))
            roots = [e for e in container.iter() if e.tag.split('}')[-1] == 'rootfile']
            preferred = [e for e in roots if e.get('media-type') == 'application/vnd.recordare.musicxml+xml']
            if not roots:
                raise ValueError('MXL has no rootfile')
            target = (preferred or roots)[0].get('full-path')
            if z.getinfo(target).file_size > 50_000_000:
                raise ValueError('MXL rootfile exceeds 50 MB')
            return z.read(target)
    return path.read_bytes()


def parse_text(xml):
    root = ET.fromstring(xml)
    for e in root.iter():
        e.tag = e.tag.split('}')[-1]
    out = dict(events=[], measures=[], tempos=[], attributes=[], errors=[], parts=[])

    def issue(code, **kw):
        out['errors'].append(dict(code=code, **kw))

    if root.tag != 'score-partwise':
        issue('unsupported_score_type', tag=root.tag)
        return out
    part_measure_lists = []
    for pi, part in enumerate(root.findall('part')):
        part_id = part.get('id', str(pi + 1))
        out['parts'].append(part_id)
        div, nominal = F(1), F(4)
        transpositions = {'all': F(0)}
        has_divisions = False
        pmeasures = []
        for mi, measure in enumerate(part.findall('measure')):
            loc = dict(part=part_id, part_index=pi, measure=measure.get('number', str(mi + 1)), measure_index=mi)
            cursor, extent, last_on = F(0), F(0), None
            for item in measure:
                if item.tag == 'attributes':
                    if item.find('divisions') is not None:
                        div = F(item.findtext('divisions'))
                        if div <= 0:
                            raise ValueError('divisions must be positive')
                        has_divisions = True
                    time = item.find('time')
                    if time is not None:
                        beats = time.findall('beats')
                        types = time.findall('beat-type')
                        if len(beats) != len(types) or not beats:
                            issue('unsupported_meter', **loc)
                        else:
                            nominal = sum((sum(F(n) for n in b.text.split('+')) * 4 / F(t.text)
                                           for b, t in zip(beats, types)), F(0))
                            out['attributes'].append(dict(**loc, kind='time', value=[(b.text, t.text) for b, t in zip(beats, types)]))
                    key = item.find('key')
                    if key is not None:
                        out['attributes'].append(dict(**loc, kind='key', fifths=key.findtext('fifths'), mode=key.findtext('mode')))
                    for tr in item.findall('transpose'):
                        if cursor != 0:
                            issue('unresolved_midmeasure_transposition', **loc, offset=cursor)
                        amount = F(tr.findtext('chromatic', '0')) + 12 * F(tr.findtext('octave-change', '0'))
                        if tr.get('number') is None:
                            transpositions = {'all': amount}
                        else:
                            transpositions[tr.get('number')] = amount
                        out['attributes'].append(dict(**loc, kind='transpose', offset=cursor, staff=tr.get('number', 'all'), semitones=amount))
                        if tr.find('double') is not None:
                            issue('unresolved_transposition_doubling', **loc)
                elif item.tag in ('backup', 'forward'):
                    amount = F(item.findtext('duration', '0')) / div
                    if amount <= 0:
                        issue('invalid_cursor_duration', **loc)
                    cursor += amount if item.tag == 'forward' else -amount
                    if cursor < 0:
                        issue('negative_cursor', **loc)
                    extent = max(extent, cursor)
                    last_on = None
                elif item.tag == 'note':
                    if any(F(item.get(k, '0')) != 0 for k in ['attack', 'release']):
                        issue('unresolved_performance_timing', **loc)
                    grace = item.find('grace') is not None
                    chord = item.find('chord') is not None
                    if not has_divisions and not grace:
                        issue('missing_divisions', **loc)
                    dur = F(item.findtext('duration', '0')) / div
                    if not grace and dur <= 0:
                        issue('invalid_note_duration', **loc)
                    if grace:
                        issue('unresolved_grace', **loc, offset=cursor)
                    onset = last_on if chord and last_on is not None else cursor
                    if chord and last_on is None:
                        issue('orphan_chord', **loc)
                    if not chord:
                        last_on = cursor
                        cursor += dur
                    extent = max(extent, cursor, onset + dur)
                    pitch = item.find('pitch')
                    if pitch is not None:
                        step = pitch.findtext('step')
                        alter = F(pitch.findtext('alter', '0'))
                        octave = int(pitch.findtext('octave'))
                        written_midi = F(12 * (octave + 1) + STEPS[step]) + alter
                        staff = item.findtext('staff', '1')
                        transposition = transpositions.get(staff, transpositions['all'])
                        midi = written_midi + transposition
                        tie = {t.get('type') for t in item.findall('tie')}
                        tied = {t.get('type') for t in item.findall('./notations/tied')}
                        if tied - {'start', 'stop'}:
                            issue('unsupported_tie_notation', **loc)
                        if tied and tied != tie:
                            issue('tie_sound_notation_mismatch', **loc)
                        if any(t.get('time-only') for t in item.findall('tie')):
                            issue('conditional_tie', **loc)
                        event = dict(**loc, offset=onset, dur=dur, midi=midi,
                                     pitch=pitch_name(midi), written=f'{step}:{alter}:{octave}',
                                     written_midi=written_midi, transposition=transposition,
                                     voice=item.findtext('voice', '1'), staff=staff,
                                     ties=tie, grace=grace, order=len(out['events']))
                        out['events'].append(event)
                    elif item.find('rest') is None:
                        issue('unpitched_or_unknown_note', **loc)
                    for tag in ['tremolo', 'trill-mark', 'mordent', 'inverted-mordent', 'turn', 'inverted-turn']:
                        if item.find(f'.//{tag}') is not None:
                            issue('unresolved_ornament', ornament=tag, **loc)
                elif item.tag in ('direction', 'sound'):
                    offset_element = item.find('offset')
                    direction_offset = F(0)
                    if offset_element is not None and (item.tag == 'sound' or offset_element.get('sound', 'no') == 'yes'):
                        direction_offset = F(offset_element.text) / div
                    at = cursor + direction_offset
                    sounds = [item] if item.tag == 'sound' else item.findall('sound')
                    for sound in sounds:
                        sound_at = cursor + F(sound.findtext('offset')) / div if sound.find('offset') is not None else at
                        if sound.get('tempo') is not None:
                            out['tempos'].append(dict(**loc, offset=sound_at, qpm=F(sound.get('tempo')), source='sound'))
                        if any(k in sound.attrib for k in ['dacapo', 'dalsegno', 'tocoda', 'fine', 'forward-repeat']):
                            issue('unexpanded_navigation', **loc)
                        if sound.get('time-only'):
                            issue('conditional_sound', **loc)
                    for met in item.findall('./direction-type/metronome'):
                        unit = met.findtext('beat-unit')
                        per = met.findtext('per-minute')
                        if unit in UNITS and per is not None:
                            try:
                                dots = len(met.findall('beat-unit-dot'))
                                multiplier = sum((F(1, 2**i) for i in range(dots + 1)), F(0))
                                beat_length = UNITS[unit] * multiplier
                                for tied_unit in met.findall('beat-unit-tied'):
                                    tied_dots = len(tied_unit.findall('beat-unit-dot'))
                                    beat_length += UNITS[tied_unit.findtext('beat-unit')] * sum((F(1, 2**i) for i in range(tied_dots + 1)), F(0))
                                qpm = F(per) * beat_length
                                out['tempos'].append(dict(**loc, offset=at, qpm=qpm, source='metronome'))
                            except (ValueError, KeyError):
                                issue('unresolved_metronome', **loc)
                        else:
                            issue('unresolved_metronome', **loc)
                    if item.find('./direction-type/octave-shift') is not None:
                        issue('unresolved_octave_shift', **loc)
                    for word in item.findall('./direction-type/words'):
                        text = ''.join(word.itertext()).strip().lower()
                        if any(term in text for term in ['d.c.', 'd.s.', 'da capo', 'dal segno', 'rit.', 'rall.', 'accel.', 'a tempo']):
                            issue('unresolved_playback_text', text=text, **loc)
                elif item.tag == 'barline':
                    if item.find('repeat') is not None or item.find('ending') is not None:
                        issue('unexpanded_navigation', **loc)
            if extent > nominal:
                issue('overfull_measure', extent=extent, nominal=nominal, **loc)
            if extent < nominal and measure.get('implicit') != 'yes':
                issue('underfull_measure', extent=extent, nominal=nominal, **loc)
            # Keep the encoded extent for diagnostic alignment. An incomplete
            # bar without implicit=yes remains an error; do not invent silence.
            length = extent if extent > 0 else nominal
            pmeasures.append(dict(**loc, actual=extent, length=length, nominal=nominal, implicit=measure.get('implicit', 'no')))
        part_measure_lists.append(pmeasures)
        out['measures'].extend(pmeasures)

    # Align parts by measure ordinal. Misaligned parts are blocked, not certified.
    starts, cumulative = [], F(0)
    count = max((len(x) for x in part_measure_lists), default=0)
    if any(len(x) != count for x in part_measure_lists):
        issue('part_measure_count_mismatch')
    for mi in range(count):
        group = [x[mi] for x in part_measure_lists if mi < len(x)]
        if len({x['length'] for x in group}) > 1:
            issue('part_measure_length_mismatch', measure_index=mi)
        if len({x['measure'] for x in group}) > 1:
            issue('part_measure_number_mismatch', measure_index=mi)
        starts.append(cumulative)
        cumulative += max(x['length'] for x in group)
    out['total_q'] = cumulative
    for collection in ['events', 'tempos', 'measures']:
        for item in out[collection]:
            item['onset'] = starts[item['measure_index']] + item.get('offset', F(0))
    return out


def audit_score(score):
    errors = list(score['errors'])
    events = sorted(score['events'], key=lambda e: (e['onset'], e['order']))
    active, strikes = {}, []
    for event in events:
        if event['midi'] not in ALLOWED:
            errors.append(dict(code='pitch_not_allowed', event=event))
        if event['grace']:
            continue
        key = (event['part'], event['voice'], event['midi'])
        continuation = False
        if 'stop' in event['ties']:
            old = active.pop(key, None)
            if old is None or old['onset'] + old['dur'] != event['onset']:
                errors.append(dict(code='orphan_tie_stop', event=event))
            else:
                continuation = True
        elif key in active:
            errors.append(dict(code='broken_tie_chain', event=event))
            active.pop(key)
        if not continuation:
            strikes.append(event)
        if 'start' in event['ties']:
            active[key] = event
    for event in active.values():
        errors.append(dict(code='unclosed_tie', event=event))
    groups = defaultdict(list)
    for e in strikes:
        groups[(e['midi'], e['onset'])].append(e)
    unique = []
    for (midi, onset), group in groups.items():
        unique.append(group[0])
        if len(group) > 1:
            errors.append(dict(code='duplicate_strike', pitch=pitch_name(midi), onset=onset, events=group))
    previous = {}
    gaps = []
    for event in sorted(unique, key=lambda e: (e['onset'], e['midi'])):
        midi = event['midi']
        if midi in previous:
            gap = event['onset'] - previous[midi]['onset']
            gaps.append(gap)
            if gap < 1:
                errors.append(dict(code='same_pitch_too_close', gap_q=gap, previous=previous[midi], event=event))
        previous[midi] = event
    tempo_groups = defaultdict(set)
    for tempo in score['tempos']:
        tempo_groups[tempo['onset']].add(tempo['qpm'])
        if not 100 <= tempo['qpm'] <= 220:
            errors.append(dict(code='tempo_out_of_range', tempo=tempo))
    if not any(t <= 0 for t in tempo_groups):
        errors.append(dict(code='missing_initial_tempo'))
    for onset, values in tempo_groups.items():
        if len(values) > 1:
            errors.append(dict(code='tempo_conflict', onset=onset, values=sorted(values)))
    if not events:
        errors.append(dict(code='empty_score'))
    pitches = [e['midi'] for e in events]
    return dict(ok=not errors, scope='written linear score; unresolved playback constructs block PASS',
                parts=len(score['parts']), measures_per_part=dict(Counter(m['part'] for m in score['measures'])),
                total_quarter_units=score.get('total_q', 0), noteheads=len(events),
                strike_noteheads=len(strikes), unique_strikes=len(unique),
                pitch_range=[pitch_name(min(pitches)), pitch_name(max(pitches))] if pitches else [],
                used_pitches={pitch_name(p): n for p, n in sorted(Counter(pitches).items())},
                strike_duration_histogram=dict(Counter(e['dur'] for e in strikes)),
                minimum_same_pitch_gap=min(gaps) if gaps else None,
                tempo_events=score['tempos'], attributes=score['attributes'],
                error_counts=dict(Counter(e['code'] for e in errors)), errors=errors)


def audit_text(xml):
    return audit_score(parse_text(xml))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('score', type=Path)
    parser.add_argument('--json', type=Path, help='Write full diagnostic JSON without editing the score')
    args = parser.parse_args()
    if args.json and args.json.resolve() == args.score.resolve():
        parser.error('--json must not overwrite the source score')
    try:
        report = audit_text(read_xml(args.score))
        report['source'] = str(args.score.resolve())
        report['sha256'] = hashlib.sha256(args.score.read_bytes()).hexdigest()
    except (ET.ParseError, ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        report = dict(ok=False, error_counts={'parse_error': 1}, errors=[dict(code='parse_error', message=str(exc))])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(serial(report), ensure_ascii=False, indent=2), encoding='utf-8')
    short = {k: v for k, v in report.items() if k not in ['errors', 'attributes', 'used_pitches', 'strike_duration_histogram', 'tempo_events']}
    print(json.dumps(serial(short), ensure_ascii=False, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
