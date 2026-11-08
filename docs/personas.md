# Penny: preset games and resident profiles

*For the pitch and the demo. The residents are **fictional composites**, built from the needs in
[evidence.md](evidence.md); say so once in the pitch ("Margaret is a composite of the residents we read about").*

## Preset games

The games Penny knows out of the box (all on `main`), and who each one suits.

| Game | In one line | Good for | Penny's role |
|---|---|---|---|
| **Noughts and crosses** | Three in a row wins | A quick first game; everyone knows it | Opponent, or writes for you |
| **Hangman** | Guess the word letter by letter before the drawing is finished | Word lovers, crossword people | Draws the dashes and the picture, writes your letters |
| **Dots and boxes** | Join two dots; finish a box to claim it and go again | People who like a bit of strategy | Opponent; draws lines and initials |
| **Backwards noughts and crosses** | Make three in a row and you *lose* | A twist on a familiar game, good for laughs | Opponent |
| **Footsteps** | Mark a square next to the last mark; if you can't move, you lose | Our teaching example ([rules](games/footsteps.pdf)) | Learns it, then explains the trick |
| **Footsteps with diagonals** | The same, diagonals allowed | A harder variant | Opponent |

Plus every game a resident teaches her, saved in the library with its win/loss record.

## Resident profiles

### Margaret, 84 · the teacher
- **Background:** retired primary school teacher. Played paper games with her pupils and her grandchildren for years.
- **Hands:** essential tremor; her writing has become shaky and she's embarrassed by it.
- **How she plays:** *"You write for me."* She says the square, Penny repeats it, then draws.
- **Favourite:** noughts and crosses with her grandson when he visits.
- **Taught Penny:** a game from her childhood (in the demo, Footsteps stands in for it).
- **Level:** gentle, moving up to fair as she wins.
- **Preferences:** a slower voice; likes Penny to use her name.
- **Her line in the pitch:** the main character: the quiet afternoon, teaching her game, the game spreading.

### Arthur, 79 · the competitor
- **Background:** retired bus driver, lives down the hall from Margaret. Bored in the afternoons; doesn't know many residents.
- **Hands:** fine. **Hearing:** a little hard of hearing, so the page shows what Penny says.
- **How she plays:** *"I'll write."* He draws his own marks; Penny reads them with the camera.
- **Favourites:** dots and boxes, hangman.
- **Level:** "play your best": he wants a proper opponent.
- **Learned from Penny:** Margaret's game. *"Margaret taught me this one. Would you like to learn it?"*
- **His line in the pitch:** the spread. Penny connects two residents who'd never played together.

### Doreen, 91 · the word lover
- **Background:** lifelong crossword solver; still sharp, tires quickly.
- **Hands:** weakness in her right hand after a stroke two years ago.
- **How she plays:** *mix.* She writes when she feels up to it and says *"you do this one"* when she's tired.
- **Favourite:** hangman.
- **Level:** fair. Short sessions.
- **Preferences:** likes Penny to read back the letters she's tried (Penny answers questions from the live game).
- **Her line in the pitch:** optional; shows the "mix" way of playing and that Penny adapts to the person.

### Priya · the carer
- **Role:** care assistant on the afternoon shift, fifteen residents.
- **Uses Penny for:** setting up a game for a resident in under a minute, then getting on with her rounds.
- **Cares about:** safety (the arm freezes when a hand comes near), privacy (nothing leaves the room), not having to babysit it.
- **Her line in the pitch:** *"Penny doesn't replace me. She covers the hour I don't have."*

## How to use them

**In the pitch.** Margaret carries the story; Arthur is the payoff; Priya gives the staff view. Doreen is optional,
for a question about different abilities ("what about someone who can write some days and not others?").

**In the demo.** Greet the profile by name, so judges see Penny remembers people:
- *"Hello Margaret. Shall I write for you again?"* (her saved preference)
- *"Arthur, Margaret taught me a new game this week. Want to learn it?"* (the library)

**In questions.** Use the profiles to answer "who is this for?" with a person, not a statistic.

## Seed data for the demo (optional, for whoever owns the brain)

If there's time, preloading these makes the demo feel lived-in. Names, way of playing and level are the useful
fields; the rest is story.

```json
[
  {"name": "Margaret", "way_to_play": "write_for_me", "level": "gentle", "voice_speed": "slow",
   "favourite": "noughts and crosses", "taught": ["Footsteps"]},
  {"name": "Arthur", "way_to_play": "i_write", "level": "best", "favourite": "dots and boxes",
   "show_captions": true},
  {"name": "Doreen", "way_to_play": "mix", "level": "fair", "favourite": "hangman", "short_sessions": true}
]
```

The fields above are a suggestion, not the code's format; persona files are being built (`penny/residents`)
(see [extras-backlog.md](extras-backlog.md)).
