
using System;
using System.Collections.Generic;

class TernarySigilInterpreter
{
    static Dictionary<string, (string Name, string Description, string Glyph)> sigilMap = new Dictionary<string, (string, string, string)>
    {
        { "T01T1", ("Mirror Spiral", "Reflection into recursion", "🌀🔁") },
        { "00001", ("Sulfur", "Symbol of desire, energy, and transformative fire", "🜔") },
        { "00010", ("Mercury", "Agent of fluidity, resonance, and mental exchange", "🜍") },
        { "0T010", ("Salt", "Stabilizer, the harmonizing principle between opposites", "🜨") },
        // Add more as needed
    };

    static Dictionary<string, string> glyphToTernary = new Dictionary<string, string>
    {
        { "🜔", "00001" },
        { "🜍", "00010" },
        { "🜨", "0T010" },
        { "🜕", "01000" },
        // Extend with full glyph mappings
    };

    static void Main()
    {
        Console.WriteLine("🔺 Balanced-Ternary Sigil Interpreter");
        Console.WriteLine("Enter a 5-digit ternary code (T, 0, 1) or glyphs (e.g. 🜔🜍🜨):");
        string input = Console.ReadLine().ToUpper();

        if (input.Length == 5 && input.IndexOfAny(new[] { 'T', '0', '1' }) != -1)
        {
            if (sigilMap.TryGetValue(input, out var sigil))
            {
                Console.WriteLine($"{sigil.Glyph} {sigil.Name}");
                Console.WriteLine($""{sigil.Description}"");
            }
            else
            {
                Console.WriteLine("Sigil not found.");
            }
        }
        else
        {
            foreach (char glyph in input)
            {
                string g = glyph.ToString();
                if (glyphToTernary.ContainsKey(g))
                {
                    Console.WriteLine($"{g} → {glyphToTernary[g]}");
                }
                else
                {
                    Console.WriteLine($"{g} → ❌ Not found");
                }
            }
        }
    }
}
