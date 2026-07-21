In this variable subset choice dataset, the variable choice set is all items
clicked on by a user in a given browsing session on an e-commerce web site. The
subset selection is the set of items purchased in that session, which is a
subset of those that were clicked on. This dataset was derived from data for
the 2015 RecSys challenge:
http://2015.recsyschallenge.com/

Each line of the two data files
1. vchoice-Yc-Items.txt
2. vchoice-Yc-Items-5-10-4-8.txt
represents a subset selection. The line is separated by a semicolon, the first
part of which is the slate of items and the second part of which is the subset
selection. For example, the line
30774 12821 3147;30774 3147
means that the choice set is {30774, 12821, 3147} and the subset selection is
{30774, 3147}.

The second data file is represents a subset of the original dataset where every
subset selection is of size at most 5, every choice set is of size at most 10,
every item is selected in a subset at least 4 times, and every item appears in a
choice set at least 8 times. The experiments in our paper uses this restricted
dataset.
